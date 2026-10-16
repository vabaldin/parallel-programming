#include "gemm.h"
#include "matmul.h"

/* One Strassen level saves 1/8 of the multiplications; below this size the
 * extra additions cost more than that. */
#ifndef BEST_STRASSEN_MIN
#define BEST_STRASSEN_MIN 4096
#endif

void matmul_best(size_t n, const float *a, const float *b, float *c) {
    if (n >= BEST_STRASSEN_MIN)
        strassen_mul(n, a, n, b, n, c, n, gemm_neon_omp, n / 2, 1);
    else
        gemm_neon_omp(n, a, n, b, n, c, n);
}
