#!/usr/bin/env python3
"""
Fuzz tests for the standalone C implementations of cvtColor and INTER_AREA resize.
The script builds libmini_cv.so with gcc, loads it via ctypes, and compares results
with OpenCV on random uint8 inputs.
"""

import ctypes
import enum
import os
import random
import subprocess
import sys
from time import perf_counter
from typing import Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
LIB_PATH = os.path.join(ROOT, "libmini_cv.so")


def build_shared_lib() -> None:
    if os.path.exists(LIB_PATH):
        return
    cmd = [
        "gcc",
        "-O2",
        "-g",
        "-fno-omit-frame-pointer",
        "-std=c99",
        "-shared",
        "-fPIC",
        os.path.join(ROOT, "mini_cv.c"),
        "-o",
        LIB_PATH,
        "-lm",
        "-lunwind",
    ]
    subprocess.check_call(cmd, cwd=ROOT)


class MiniColor(enum.IntEnum):
    BGR2GRAY = 0
    RGB2GRAY = 1
    GRAY2BGR = 2
    GRAY2RGB = 3
    BGR2RGB = 4
    RGB2BGR = 5


def _load_lib():
    build_shared_lib()
    lib = ctypes.CDLL(LIB_PATH)
    lib.mini_cvtcolor_u8.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mini_cvtcolor_u8.restype = ctypes.c_int

    lib.mini_resize_area_u8.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mini_resize_area_u8.restype = ctypes.c_int
    return lib


LIB = _load_lib()


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def run_cvtcolor(src: np.ndarray, code: MiniColor, dcn: int) -> np.ndarray:
    h, w = src.shape[:2]
    scn = 1 if src.ndim == 2 else src.shape[2]
    dst_shape = (h, w, dcn) if dcn > 1 else (h, w)
    dst = np.empty(dst_shape, dtype=np.uint8)
    ret = LIB.mini_cvtcolor_u8(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w,
        h,
        src.strides[0],
        scn,
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.strides[0],
        dcn,
        int(code),
    )
    if ret != 0:
        raise RuntimeError(f"mini_cvtcolor_u8 failed with code {ret}")
    return dst


def run_resize(src: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    h, w = src.shape[:2]
    scn = 1 if src.ndim == 2 else src.shape[2]
    dst_w, dst_h = new_size
    dst_shape = (dst_h, dst_w, scn) if scn > 1 else (dst_h, dst_w)
    dst = np.empty(dst_shape, dtype=np.uint8)
    ret = LIB.mini_resize_area_u8(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w,
        h,
        src.strides[0],
        scn,
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst_w,
        dst_h,
        dst.strides[0],
    )
    if ret != 0:
        raise RuntimeError(f"mini_resize_area_u8 failed with code {ret}")
    return dst


def fuzz_color(iterations: int) -> int:
    failures = 0
    ops = [
        (MiniColor.BGR2GRAY, cv2.COLOR_BGR2GRAY, 3, 1),
        (MiniColor.RGB2GRAY, cv2.COLOR_RGB2GRAY, 3, 1),
        (MiniColor.GRAY2BGR, cv2.COLOR_GRAY2BGR, 1, 3),
        (MiniColor.GRAY2RGB, cv2.COLOR_GRAY2RGB, 1, 3),
        (MiniColor.BGR2RGB, cv2.COLOR_BGR2RGB, 3, 3),
        (MiniColor.RGB2BGR, cv2.COLOR_RGB2BGR, 3, 3),
    ]
    for i in range(iterations):
        code, cv_code, scn, dcn = random.choice(ops)
        h = random.randint(1, 64)
        w = random.randint(1, 64)
        shape = (h, w) if scn == 1 else (h, w, scn)
        src = _as_uint8(np.random.randint(0, 256, shape, dtype=np.uint8))

        ref = cv2.cvtColor(src, cv_code)
        got = run_cvtcolor(src, code, dcn)

        if not np.array_equal(ref, got):
            failures += 1
            diff = np.abs(ref.astype(np.int16) - got.astype(np.int16))
            print(f"[color][{i}] mismatch for {code.name}, max diff {diff.max()}", file=sys.stderr)
    return failures


def fuzz_resize(iterations: int) -> int:
    failures = 0
    mini_time = 0.0
    cv_time = 0.0
    max_w, max_h = 1280, 720  # up to 720p
    for i in range(iterations):
        scn = random.choice([1, 3, 4])
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        # Only test true downscale or keep-size so we match OpenCV's INTER_AREA path.
        dst_h = random.randint(1, h)
        dst_w = random.randint(1, w)
        shape = (h, w) if scn == 1 else (h, w, scn)
        src = _as_uint8(np.random.randint(0, 256, shape, dtype=np.uint8))

        t0 = perf_counter()
        ref = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        cv_time += perf_counter() - t0

        t1 = perf_counter()
        got = run_resize(src, (dst_w, dst_h))
        mini_time += perf_counter() - t1

        diff = np.abs(ref.astype(np.int16) - got.astype(np.int16))
        if diff.max() > 1:
            failures += 1
            print(
                f"[resize][{i}] max diff {diff.max()} for src {w}x{h} -> {dst_w}x{dst_h} ch {scn}",
                file=sys.stderr,
            )
    # Performance: keep within 10x of OpenCV.
    if cv_time > 0 and mini_time > cv_time * 10:
        print(
            f"[perf] mini resize slower than OpenCV by {mini_time / cv_time:.2f}x "
            f"(mini {mini_time:.4f}s, cv {cv_time:.4f}s)",
            file=sys.stderr,
        )
        failures += 1
    return failures


def main():
    random.seed(0)
    np.random.seed(0)
    color_iters = 400
    resize_iters = 400
    color_fail = fuzz_color(color_iters)
    resize_fail = fuzz_resize(resize_iters)
    if color_fail or resize_fail:
        print(f"Done with {color_fail} color failures and {resize_fail} resize failures.")
        sys.exit(1)
    print(f"All good: {color_iters} color + {resize_iters} resize cases matched.")


if __name__ == "__main__":
    main()
