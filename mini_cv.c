#include "mini_cv.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MINI_GRAY_SHIFT 15
#define MINI_RY15 9798   /* R2YF * 32768 + 0.5 */
#define MINI_GY15 19235  /* G2YF * 32768 + 0.5 */
#define MINI_BY15 3735   /* B2YF * 32768 + 0.5 */

static inline uint8_t mini_saturate_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static inline uint8_t mini_saturate_from_float(float v) {
    int iv = (int)lroundf(v);
    return mini_saturate_u8(iv);
}

struct mini_decimate_alpha {
    int si;
    int di;
    float alpha;
};

static int mini_compute_resize_area_tab(int ssize, int dsize, int cn, double scale,
                                        struct mini_decimate_alpha* tab) {
    int k = 0;
    for (int dx = 0; dx < dsize; ++dx) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cell_width = scale < (ssize - fsx1) ? scale : (ssize - fsx1);

        int sx1 = (int)ceil(fsx1);
        int sx2 = (int)floor(fsx2);

        if (sx2 > ssize - 1) sx2 = ssize - 1;
        if (sx1 > sx2) sx1 = sx2;

        if (sx1 - fsx1 > 1e-3) {
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k].alpha = (float)((sx1 - fsx1) / cell_width);
            ++k;
        }

        for (int sx = sx1; sx < sx2; ++sx) {
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k].alpha = (float)(1.0 / cell_width);
            ++k;
        }

        if (fsx2 - sx2 > 1e-3) {
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            double w = fsx2 - sx2;
            if (w > 1.0) w = 1.0;
            if (w > cell_width) w = cell_width;
            tab[k].alpha = (float)(w / cell_width);
            ++k;
        }
    }
    return k;
}

static int mini_resize_area_down(const uint8_t* src, int src_w, int src_h, int src_stride,
                                 int cn, uint8_t* dst, int dst_w, int dst_h, int dst_stride) {
    double scale_x = (double)src_w / (double)dst_w;
    double scale_y = (double)src_h / (double)dst_h;

    int xtab_capacity = src_w * 2 + 2;
    int ytab_capacity = src_h * 2 + 2;
    struct mini_decimate_alpha* xtab =
        (struct mini_decimate_alpha*)malloc(sizeof(struct mini_decimate_alpha) * xtab_capacity);
    struct mini_decimate_alpha* ytab =
        (struct mini_decimate_alpha*)malloc(sizeof(struct mini_decimate_alpha) * ytab_capacity);
    int* tabofs = (int*)malloc(sizeof(int) * (dst_h + 1));
    float* sum = (float*)malloc(sizeof(float) * dst_w * cn);
    float* buf = (float*)malloc(sizeof(float) * dst_w * cn);

    if (!xtab || !ytab || !tabofs || !sum || !buf) {
        free(xtab);
        free(ytab);
        free(tabofs);
        free(sum);
        free(buf);
        return -1;
    }

    int xtab_size = mini_compute_resize_area_tab(src_w, dst_w, cn, scale_x, xtab);
    int ytab_size = mini_compute_resize_area_tab(src_h, dst_h, 1, scale_y, ytab);

    int dy_count = 0;
    int prev_di = -1;
    for (int k = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != prev_di) {
            tabofs[dy_count++] = k;
            prev_di = ytab[k].di;
        }
    }
    tabofs[dy_count] = ytab_size;

    for (int dy = 0; dy < dst_h; ++dy) {
        memset(sum, 0, sizeof(float) * dst_w * cn);
        int y_start = tabofs[dy];
        int y_end = tabofs[dy + 1];
        for (int jy = y_start; jy < y_end; ++jy) {
            int sy = ytab[jy].si;
            float beta = ytab[jy].alpha;
            const uint8_t* srow = src + sy * src_stride;
            memset(buf, 0, sizeof(float) * dst_w * cn);
            for (int k = 0; k < xtab_size; ++k) {
                int dx = xtab[k].di;
                int sx = xtab[k].si;
                float alpha = xtab[k].alpha * beta;
                const uint8_t* sp = srow + sx;
                float* bp = buf + dx;
                for (int c = 0; c < cn; ++c) {
                    bp[c] += alpha * (float)sp[c];
                }
            }
            for (int i = 0; i < dst_w * cn; ++i) {
                sum[i] += buf[i];
            }
        }

        uint8_t* drow = dst + dy * dst_stride;
        for (int i = 0; i < dst_w * cn; ++i) {
            drow[i] = mini_saturate_from_float(sum[i]);
        }
    }

    free(xtab);
    free(ytab);
    free(tabofs);
    free(sum);
    free(buf);
    return 0;
}

static int mini_resize_area_linear(const uint8_t* src, int src_w, int src_h, int src_stride,
                                   int cn, uint8_t* dst, int dst_w, int dst_h, int dst_stride) {
    const int COEF_BITS = 11;
    const int ONE = 1 << COEF_BITS;
    double scale_x = (double)src_w / (double)dst_w;
    double scale_y = (double)src_h / (double)dst_h;
    double inv_scale_x = 1.0 / scale_x;
    double inv_scale_y = 1.0 / scale_y;
    int width = dst_w * cn;
    int* alpha = (int*)malloc(sizeof(int) * width * 2);
    int* beta = (int*)malloc(sizeof(int) * dst_h * 2);
    int* xofs = (int*)malloc(sizeof(int) * width);
    int* yofs = (int*)malloc(sizeof(int) * dst_h);
    if (!alpha || !beta || !xofs || !yofs) {
        free(alpha);
        free(beta);
        free(xofs);
        free(yofs);
        return -1;
    }

    for (int dx = 0; dx < dst_w; ++dx) {
        int sx = (int)floor(dx * scale_x);
        double fx = (dx + 1) - (sx + 1) * inv_scale_x;
        if (fx <= 0) {
            fx = 0.0;
        } else {
            fx -= floor(fx);
        }
        if (sx < 0) {
            sx = 0;
            fx = 0.0;
        }
        if (sx >= src_w - 1) {
            sx = src_w - 2;
            fx = 1.0;
        }
        int w1 = (int)lround(fx * ONE);
        int w0 = ONE - w1;
        for (int c = 0; c < cn; ++c) {
            int ofs = dx * cn + c;
            xofs[ofs] = sx * cn + c;
            alpha[ofs * 2] = w0;
            alpha[ofs * 2 + 1] = w1;
        }
    }

    for (int dy = 0; dy < dst_h; ++dy) {
        int sy = (int)floor(dy * scale_y);
        double fy = (dy + 1) - (sy + 1) * inv_scale_y;
        if (fy <= 0) {
            fy = 0.0;
        } else {
            fy -= floor(fy);
        }
        if (sy < 0) {
            sy = 0;
            fy = 0.0;
        }
        if (sy >= src_h - 1) {
            sy = src_h - 2;
            fy = 1.0;
        }
        int w1 = (int)lround(fy * ONE);
        int w0 = ONE - w1;
        yofs[dy] = sy;
        beta[dy * 2] = w0;
        beta[dy * 2 + 1] = w1;
    }

    for (int dy = 0; dy < dst_h; ++dy) {
        const uint8_t* srow0 = src + yofs[dy] * src_stride;
        const uint8_t* srow1 = src + (yofs[dy] + 1) * src_stride;
        uint8_t* drow = dst + dy * dst_stride;
        int wy0 = beta[dy * 2];
        int wy1 = beta[dy * 2 + 1];
        for (int dx = 0; dx < dst_w; ++dx) {
            int base = dx * cn;
            for (int c = 0; c < cn; ++c) {
                int ofs = base + c;
                int sx0 = xofs[ofs];
                int sx1 = sx0 + cn;
                if (sx1 >= src_w * cn) sx1 = sx0;
                int wx0 = alpha[ofs * 2];
                int wx1 = alpha[ofs * 2 + 1];
                int t0 = (wx0 * (int)srow0[sx0] + wx1 * (int)srow0[sx1] + (ONE >> 1)) >> COEF_BITS;
                int t1 = (wx0 * (int)srow1[sx0] + wx1 * (int)srow1[sx1] + (ONE >> 1)) >> COEF_BITS;
                int v = (wy0 * t0 + wy1 * t1 + (ONE >> 1)) >> COEF_BITS;
                drow[ofs] = mini_saturate_u8(v);
            }
        }
    }

    free(alpha);
    free(beta);
    free(xofs);
    free(yofs);
    return 0;
}

int mini_resize_area_u8(const uint8_t* src, int src_w, int src_h, int src_stride,
                        int channels, uint8_t* dst, int dst_w, int dst_h, int dst_stride) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0)
        return -1;

    double scale_x = (double)src_w / (double)dst_w;
    double scale_y = (double)src_h / (double)dst_h;

    if (scale_x >= 1.0 && scale_y >= 1.0) {
        return mini_resize_area_down(src, src_w, src_h, src_stride, channels, dst, dst_w, dst_h,
                                     dst_stride);
    }
    // OpenCV switches to the linear kernel when either axis is upscaled.
    return mini_resize_area_linear(src, src_w, src_h, src_stride, channels, dst, dst_w, dst_h,
                                   dst_stride);
}

static void mini_gray_to_rgb(const uint8_t* src, int width, int height, int src_stride,
                             uint8_t* dst, int dst_stride, int dcn) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* srow = src + y * src_stride;
        uint8_t* drow = dst + y * dst_stride;
        for (int x = 0; x < width; ++x) {
            uint8_t g = srow[x];
            int base = x * dcn;
            drow[base] = g;
            drow[base + 1] = g;
            drow[base + 2] = g;
            if (dcn == 4) drow[base + 3] = 255;
        }
    }
}

static void mini_bgr_to_gray(const uint8_t* src, int width, int height, int src_stride, int scn,
                             int blue_idx, uint8_t* dst, int dst_stride) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* srow = src + y * src_stride;
        uint8_t* drow = dst + y * dst_stride;
        for (int x = 0; x < width; ++x) {
            const uint8_t* p = srow + x * scn;
            int b = p[blue_idx];
            int g = p[1];
            int r = p[(blue_idx == 0) ? 2 : 0];
            int yv = (b * MINI_BY15 + g * MINI_GY15 + r * MINI_RY15 + (1 << (MINI_GRAY_SHIFT - 1))) >>
                     MINI_GRAY_SHIFT;
            drow[x] = mini_saturate_u8(yv);
        }
    }
}

static void mini_swap_rb(const uint8_t* src, int width, int height, int src_stride, int scn,
                         uint8_t* dst, int dst_stride, int dcn) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* srow = src + y * src_stride;
        uint8_t* drow = dst + y * dst_stride;
        for (int x = 0; x < width; ++x) {
            const uint8_t* p = srow + x * scn;
            int base = x * dcn;
            drow[base] = p[2];
            drow[base + 1] = p[1];
            drow[base + 2] = p[0];
            if (dcn == 4) drow[base + 3] = (scn == 4) ? p[3] : 255;
        }
    }
}

int mini_cvtcolor_u8(const uint8_t* src, int width, int height, int src_stride,
                     int src_channels, uint8_t* dst, int dst_stride, int dst_channels,
                     mini_color_code code) {
    if (!src || !dst || width <= 0 || height <= 0) return -1;

    switch (code) {
    case MINI_BGR2GRAY:
        if (src_channels < 3 || dst_channels != 1) return -1;
        mini_bgr_to_gray(src, width, height, src_stride, src_channels, 0, dst, dst_stride);
        return 0;
    case MINI_RGB2GRAY:
        if (src_channels < 3 || dst_channels != 1) return -1;
        mini_bgr_to_gray(src, width, height, src_stride, src_channels, 2, dst, dst_stride);
        return 0;
    case MINI_GRAY2BGR:
        if (src_channels != 1 || (dst_channels != 3 && dst_channels != 4)) return -1;
        mini_gray_to_rgb(src, width, height, src_stride, dst, dst_stride, dst_channels);
        return 0;
    case MINI_GRAY2RGB:
        if (src_channels != 1 || (dst_channels != 3 && dst_channels != 4)) return -1;
        mini_gray_to_rgb(src, width, height, src_stride, dst, dst_stride, dst_channels);
        return 0;
    case MINI_BGR2RGB:
        if (src_channels < 3 || (dst_channels != 3 && dst_channels != 4)) return -1;
        mini_swap_rb(src, width, height, src_stride, src_channels, dst, dst_stride, dst_channels);
        return 0;
    case MINI_RGB2BGR:
        if (src_channels < 3 || (dst_channels != 3 && dst_channels != 4)) return -1;
        mini_swap_rb(src, width, height, src_stride, src_channels, dst, dst_stride, dst_channels);
        return 0;
    default:
        return -1;
    }
}
