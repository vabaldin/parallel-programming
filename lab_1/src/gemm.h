#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

/* C = A * B on n x n submatrices with row strides lda, ldb and ldc. */
typedef void (*gemm_fn)(size_t n, const float *a, size_t lda, const float *b,
                        size_t ldb, float *c, size_t ldc);

void gemm_blocked(size_t n, const float *a, size_t lda, const float *b,
                  size_t ldb, float *c, size_t ldc);
void gemm_neon(size_t n, const float *a, size_t lda, const float *b,
               size_t ldb, float *c, size_t ldc);
void gemm_neon_omp(size_t n, const float *a, size_t lda, const float *b,
                   size_t ldb, float *c, size_t ldc);

/* Recurses until the size is at most cutoff, then calls base. Sizes that do
 * not split evenly down to the cutoff are zero-padded. With parallel set,
 * the matrix additions run in OpenMP loops. */
void strassen_mul(size_t n, const float *a, size_t lda, const float *b,
                  size_t ldb, float *c, size_t ldc, gemm_fn base,
                  size_t cutoff, int parallel);

/* Aborts on failure. */
float *scratch_alloc(size_t count);

#endif
