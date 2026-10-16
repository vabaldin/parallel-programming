#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

#include "gemm.h"
#include "matmul.h"

/* The 8 x 12 micro-tile of C lives in 24 of the 32 NEON registers. A KC x NR
 * panel of B (24 KiB) stays in L1 while the MC x KC block of A (128 KiB)
 * comes from L2. MC and NC are multiples of MR and NR, and an MC x NC tile
 * of C is one unit of parallel work. The values were tuned on the M3 Pro. */
#define MR 8
#define NR 12
#ifndef KC
#define KC 512
#endif
#ifndef MC
#define MC 64
#endif
#ifndef NC
#define NC 192
#endif

static size_t min_size(size_t x, size_t y) { return x < y ? x : y; }

#define LOAD_ROW(r)                                                           \
    float32x4_t c##r##0 = accumulate ? vld1q_f32(c + r * ldc) : zero;         \
    float32x4_t c##r##1 = accumulate ? vld1q_f32(c + r * ldc + 4) : zero;     \
    float32x4_t c##r##2 = accumulate ? vld1q_f32(c + r * ldc + 8) : zero;

#define FMA_ROW(r, av, lane)                                                  \
    c##r##0 = vfmaq_laneq_f32(c##r##0, b0, av, lane);                         \
    c##r##1 = vfmaq_laneq_f32(c##r##1, b1, av, lane);                         \
    c##r##2 = vfmaq_laneq_f32(c##r##2, b2, av, lane);

#define STORE_ROW(r)                                                          \
    vst1q_f32(c + r * ldc, c##r##0);                                          \
    vst1q_f32(c + r * ldc + 4, c##r##1);                                      \
    vst1q_f32(c + r * ldc + 8, c##r##2);

/* C[8 x 12] (+)= A panel * B panel; a holds MR floats per k, b holds NR. */
static void kernel(size_t kc, const float *restrict a, const float *restrict b,
                   float *restrict c, size_t ldc, int accumulate) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    LOAD_ROW(0) LOAD_ROW(1) LOAD_ROW(2) LOAD_ROW(3)
    LOAD_ROW(4) LOAD_ROW(5) LOAD_ROW(6) LOAD_ROW(7)

    for (size_t k = 0; k < kc; ++k) {
        float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4),
                    b2 = vld1q_f32(b + 8);
        FMA_ROW(0, a0, 0) FMA_ROW(1, a0, 1) FMA_ROW(2, a0, 2)
        FMA_ROW(3, a0, 3) FMA_ROW(4, a1, 0) FMA_ROW(5, a1, 1)
        FMA_ROW(6, a1, 2) FMA_ROW(7, a1, 3)
        a += MR;
        b += NR;
    }

    STORE_ROW(0) STORE_ROW(1) STORE_ROW(2) STORE_ROW(3)
    STORE_ROW(4) STORE_ROW(5) STORE_ROW(6) STORE_ROW(7)
}

/* Partial tiles at the right and bottom edges go through a full-size buffer. */
static void kernel_edge(size_t kc, const float *a, const float *b, float *c,
                        size_t ldc, size_t rows, size_t cols, int accumulate) {
    float buf[MR * NR];
    if (accumulate)
        for (size_t i = 0; i < rows; ++i)
            memcpy(buf + i * NR, c + i * ldc, cols * sizeof(float));
    kernel(kc, a, b, buf, NR, accumulate);
    for (size_t i = 0; i < rows; ++i)
        memcpy(c + i * ldc, buf + i * NR, cols * sizeof(float));
}

/* Row panel p of A: element (k, r) at p * n * MR + k * MR + r. Rows past n
 * are zero. */
static void pack_a(size_t n, const float *a, size_t lda, size_t p,
                   float *dst) {
    size_t i0 = p * MR, rows = min_size(MR, n - i0);
    dst += p * n * MR;
    for (size_t k = 0; k < n; ++k) {
        size_t r = 0;
        for (; r < rows; ++r)
            dst[k * MR + r] = a[(i0 + r) * lda + k];
        for (; r < MR; ++r)
            dst[k * MR + r] = 0.0f;
    }
}

/* Column panel q of B: element (k, j) at q * n * NR + k * NR + j. */
static void pack_b(size_t n, const float *b, size_t ldb, size_t q,
                   float *dst) {
    size_t j0 = q * NR, cols = min_size(NR, n - j0);
    dst += q * n * NR;
    for (size_t k = 0; k < n; ++k) {
        memcpy(dst + k * NR, b + k * ldb + j0, cols * sizeof(float));
        memset(dst + k * NR + cols, 0, (NR - cols) * sizeof(float));
    }
}

static void tile(size_t n, size_t it, size_t jt, const float *ap,
                 const float *bp, float *c, size_t ldc) {
    size_t i0 = it * MC, i1 = min_size(i0 + MC, n);
    size_t j0 = jt * NC, j1 = min_size(j0 + NC, n);
    for (size_t k0 = 0; k0 < n; k0 += KC) {
        size_t kc = min_size(KC, n - k0);
        for (size_t j = j0; j < j1; j += NR) {
            const float *bk = bp + (j / NR) * n * NR + k0 * NR;
            size_t cols = min_size(NR, n - j);
            for (size_t i = i0; i < i1; i += MR) {
                const float *ak = ap + (i / MR) * n * MR + k0 * MR;
                size_t rows = min_size(MR, n - i);
                float *cij = c + i * ldc + j;
                if (rows == MR && cols == NR)
                    kernel(kc, ak, bk, cij, ldc, k0 > 0);
                else
                    kernel_edge(kc, ak, bk, cij, ldc, rows, cols, k0 > 0);
            }
        }
    }
}

static void gemm(size_t n, const float *a, size_t lda, const float *b,
                 size_t ldb, float *c, size_t ldc, int par) {
    if (n == 0)
        return;
    size_t mp = (n + MR - 1) / MR, np = (n + NR - 1) / NR;
    size_t mt = (n + MC - 1) / MC, nt = (n + NC - 1) / NC;
    float *ap = scratch_alloc(mp * MR * n), *bp = scratch_alloc(np * NR * n);

    /* Dynamic scheduling lets the performance cores take more tiles than the
     * efficiency cores. */
#pragma omp parallel if (par)
    {
#pragma omp for schedule(static)
        for (size_t p = 0; p < mp; ++p)
            pack_a(n, a, lda, p, ap);
#pragma omp for schedule(static)
        for (size_t q = 0; q < np; ++q)
            pack_b(n, b, ldb, q, bp);
#pragma omp for schedule(dynamic, 1) collapse(2)
        for (size_t it = 0; it < mt; ++it)
            for (size_t jt = 0; jt < nt; ++jt)
                tile(n, it, jt, ap, bp, c, ldc);
    }

    free(ap);
    free(bp);
}

void gemm_neon(size_t n, const float *a, size_t lda, const float *b,
               size_t ldb, float *c, size_t ldc) {
    gemm(n, a, lda, b, ldb, c, ldc, 0);
}

void gemm_neon_omp(size_t n, const float *a, size_t lda, const float *b,
                   size_t ldb, float *c, size_t ldc) {
    gemm(n, a, lda, b, ldb, c, ldc, 1);
}

void matmul_simd(size_t n, const float *a, const float *b, float *c) {
    gemm_neon(n, a, n, b, n, c, n);
}

void matmul_simd_omp(size_t n, const float *a, const float *b, float *c) {
    gemm_neon_omp(n, a, n, b, n, c, n);
}
