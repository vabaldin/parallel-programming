#ifndef LAB_1_SRC_GEMM_H_
#define LAB_1_SRC_GEMM_H_

#include <cstddef>

namespace matmul {

// C = A * B on n x n submatrices with row strides lda, ldb and ldc.
using GemmFn = void (*)(std::size_t n, const float* a, std::size_t lda,
                        const float* b, std::size_t ldb, float* c,
                        std::size_t ldc);

void GemmBlocked(std::size_t n, const float* a, std::size_t lda, const float* b,
                 std::size_t ldb, float* c, std::size_t ldc);
void GemmNeon(std::size_t n, const float* a, std::size_t lda, const float* b,
              std::size_t ldb, float* c, std::size_t ldc);
void GemmNeonOmp(std::size_t n, const float* a, std::size_t lda, const float* b,
                 std::size_t ldb, float* c, std::size_t ldc);

// Recurses until the size is at most cutoff, then calls base. Sizes that do
// not split evenly down to the cutoff are zero-padded. With parallel set,
// the matrix additions run in OpenMP loops.
void StrassenMul(std::size_t n, const float* a, std::size_t lda, const float* b,
                 std::size_t ldb, float* c, std::size_t ldc, GemmFn base,
                 std::size_t cutoff, bool parallel);

}  // namespace matmul

#endif  // LAB_1_SRC_GEMM_H_
