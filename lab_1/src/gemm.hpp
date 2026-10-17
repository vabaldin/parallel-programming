#pragma once

#include <cstddef>

namespace matmul {

// C = A * B on n x n submatrices with row strides lda, ldb and ldc.
using GemmFn = void (*)(std::size_t n, const float *a, std::size_t lda,
                        const float *b, std::size_t ldb, float *c,
                        std::size_t ldc);

void gemm_blocked(std::size_t n, const float *a, std::size_t lda,
                  const float *b, std::size_t ldb, float *c, std::size_t ldc);
void gemm_neon(std::size_t n, const float *a, std::size_t lda, const float *b,
               std::size_t ldb, float *c, std::size_t ldc);
void gemm_neon_omp(std::size_t n, const float *a, std::size_t lda,
                   const float *b, std::size_t ldb, float *c, std::size_t ldc);

// Recurses until the size is at most cutoff, then calls base. Sizes that do
// not split evenly down to the cutoff are zero-padded. With parallel set,
// the matrix additions run in OpenMP loops.
void strassen_mul(std::size_t n, const float *a, std::size_t lda,
                  const float *b, std::size_t ldb, float *c, std::size_t ldc,
                  GemmFn base, std::size_t cutoff, bool parallel);

} // namespace matmul
