#include <string.h>

#include "gemm.h"
#include "matmul.h"

/* Three 64 x 64 float tiles take 48 KiB and fit in the 64 KiB L1d of the
 * smallest core. */
#ifndef BLOCK
#define BLOCK 64
#endif

static size_t min_size(size_t x, size_t y) { return x < y ? x : y; }

static void block_row(size_t n, size_t ii, const float *a, size_t lda,
                      const float *b, size_t ldb, float *c, size_t ldc) {
    size_t i1 = min_size(ii + BLOCK, n);
    for (size_t i = ii; i < i1; ++i)
        memset(c + i * ldc, 0, n * sizeof(float));

    for (size_t kk = 0; kk < n; kk += BLOCK) {
        size_t k1 = min_size(kk + BLOCK, n);
        for (size_t jj = 0; jj < n; jj += BLOCK) {
            size_t j1 = min_size(jj + BLOCK, n);
            for (size_t i = ii; i < i1; ++i) {
                float *ci = c + i * ldc;
                for (size_t k = kk; k < k1; ++k) {
                    float aik = a[i * lda + k];
                    const float *bk = b + k * ldb;
                    for (size_t j = jj; j < j1; ++j)
                        ci[j] += aik * bk[j];
                }
            }
        }
    }
}

void gemm_blocked(size_t n, const float *a, size_t lda, const float *b,
                  size_t ldb, float *c, size_t ldc) {
    for (size_t ii = 0; ii < n; ii += BLOCK)
        block_row(n, ii, a, lda, b, ldb, c, ldc);
}

void matmul_blocked(size_t n, const float *a, const float *b, float *c) {
    gemm_blocked(n, a, n, b, n, c, n);
}

/* Each thread owns whole block rows of C, so threads share a cache line of C
 * only at the boundary of two block rows. */
void matmul_blocked_omp(size_t n, const float *a, const float *b, float *c) {
    size_t nblocks = (n + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t bi = 0; bi < nblocks; ++bi)
        block_row(n, bi * BLOCK, a, n, b, n, c, n);
}
