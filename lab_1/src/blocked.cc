#include <algorithm>

#include "gemm.h"
#include "matmul.h"

namespace matmul {

namespace {

// Three 64 x 64 float tiles take 48 KiB and fit in the 64 KiB L1d of the
// smallest core.
constexpr std::size_t kBlock = 64;

void BlockRow(std::size_t n, std::size_t ii, const float* a, std::size_t lda,
              const float* b, std::size_t ldb, float* c, std::size_t ldc) {
  std::size_t i1 = std::min(ii + kBlock, n);
  for (std::size_t i = ii; i < i1; ++i) std::fill_n(c + i * ldc, n, 0.0f);

  for (std::size_t kk = 0; kk < n; kk += kBlock) {
    std::size_t k1 = std::min(kk + kBlock, n);
    for (std::size_t jj = 0; jj < n; jj += kBlock) {
      std::size_t j1 = std::min(jj + kBlock, n);
      for (std::size_t i = ii; i < i1; ++i) {
        float* ci = c + i * ldc;
        for (std::size_t k = kk; k < k1; ++k) {
          float aik = a[i * lda + k];
          const float* bk = b + k * ldb;
          for (std::size_t j = jj; j < j1; ++j) ci[j] += aik * bk[j];
        }
      }
    }
  }
}

}  // namespace

void GemmBlocked(std::size_t n, const float* a, std::size_t lda, const float* b,
                 std::size_t ldb, float* c, std::size_t ldc) {
  for (std::size_t ii = 0; ii < n; ii += kBlock) {
    BlockRow(n, ii, a, lda, b, ldb, c, ldc);
  }
}

void Blocked(std::size_t n, const float* a, const float* b, float* c) {
  GemmBlocked(n, a, n, b, n, c, n);
}

// Each thread owns whole block rows of C, so threads share a cache line of C
// only at the boundary of two block rows.
void BlockedOmp(std::size_t n, const float* a, const float* b, float* c) {
  std::size_t nblocks = (n + kBlock - 1) / kBlock;
#pragma omp parallel for schedule(dynamic, 1)
  for (std::size_t bi = 0; bi < nblocks; ++bi) {
    BlockRow(n, bi * kBlock, a, n, b, n, c, n);
  }
}

}  // namespace matmul
