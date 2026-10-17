#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <utility>

#include "gemm.h"
#include "matmul.h"

namespace matmul {

namespace {

// The 8 x 12 micro-tile of C lives in 24 of the 32 NEON registers. A KC x NR
// panel of B (24 KiB) stays in L1 while the MC x KC block of A (128 KiB)
// comes from L2. MC and NC are multiples of MR and NR, and an MC x NC tile
// of C is one unit of parallel work. The values were tuned on the M3 Pro.
constexpr std::size_t kKc = 512;
constexpr std::size_t kMc = 64;
constexpr std::size_t kNc = 192;
constexpr std::size_t kMr = 8;
constexpr std::size_t kNr = 12;

using RegisterTile = std::array<std::array<float32x4_t, 3>, kMr>;

// Row R of the tile += A[R] * B row; the lane index must be a constant.
template <std::size_t R>
void FmaRow(RegisterTile& acc, float32x4_t a_lo, float32x4_t a_hi,
            float32x4_t b0, float32x4_t b1, float32x4_t b2) {
  constexpr int kLane = R % 4;
  float32x4_t a = R < 4 ? a_lo : a_hi;
  acc[R][0] = vfmaq_laneq_f32(acc[R][0], b0, a, kLane);
  acc[R][1] = vfmaq_laneq_f32(acc[R][1], b1, a, kLane);
  acc[R][2] = vfmaq_laneq_f32(acc[R][2], b2, a, kLane);
}

template <std::size_t... R>
void FmaRows(RegisterTile& acc, float32x4_t a_lo, float32x4_t a_hi,
             float32x4_t b0, float32x4_t b1, float32x4_t b2,
             std::index_sequence<R...>) {
  (FmaRow<R>(acc, a_lo, a_hi, b0, b1, b2), ...);
}

// C[8 x 12] (+)= A panel * B panel; a holds kMr floats per k, b holds kNr.
void Kernel(std::size_t kc, const float* __restrict a,
            const float* __restrict b, float* __restrict c, std::size_t ldc,
            bool accumulate) {
  RegisterTile acc;
  for (std::size_t r = 0; r < kMr; ++r) {
    for (std::size_t j = 0; j < 3; ++j) {
      acc[r][j] =
          accumulate ? vld1q_f32(c + r * ldc + 4 * j) : vdupq_n_f32(0.0f);
    }
  }
  for (std::size_t k = 0; k < kc; ++k) {
    float32x4_t a_lo = vld1q_f32(a), a_hi = vld1q_f32(a + 4);
    float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
    FmaRows(acc, a_lo, a_hi, b0, b1, b2, std::make_index_sequence<kMr>());
    a += kMr;
    b += kNr;
  }
  for (std::size_t r = 0; r < kMr; ++r) {
    for (std::size_t j = 0; j < 3; ++j) {
      vst1q_f32(c + r * ldc + 4 * j, acc[r][j]);
    }
  }
}

// Partial tiles at the right and bottom edges go through a full-size buffer.
void KernelEdge(std::size_t kc, const float* a, const float* b, float* c,
                std::size_t ldc, std::size_t rows, std::size_t cols,
                bool accumulate) {
  float buf[kMr * kNr];
  if (accumulate) {
    for (std::size_t i = 0; i < rows; ++i) {
      std::copy_n(c + i * ldc, cols, buf + i * kNr);
    }
  }
  Kernel(kc, a, b, buf, kNr, accumulate);
  for (std::size_t i = 0; i < rows; ++i) {
    std::copy_n(buf + i * kNr, cols, c + i * ldc);
  }
}

// Row panel p of A: element (k, r) at p * n * kMr + k * kMr + r. Rows past n
// are zero.
void PackA(std::size_t n, const float* a, std::size_t lda, std::size_t p,
           float* dst) {
  std::size_t i0 = p * kMr, rows = std::min(kMr, n - i0);
  dst += p * n * kMr;
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t r = 0;
    for (; r < rows; ++r) dst[k * kMr + r] = a[(i0 + r) * lda + k];
    for (; r < kMr; ++r) dst[k * kMr + r] = 0.0f;
  }
}

// Column panel q of B: element (k, j) at q * n * kNr + k * kNr + j.
void PackB(std::size_t n, const float* b, std::size_t ldb, std::size_t q,
           float* dst) {
  std::size_t j0 = q * kNr, cols = std::min(kNr, n - j0);
  dst += q * n * kNr;
  for (std::size_t k = 0; k < n; ++k) {
    std::copy_n(b + k * ldb + j0, cols, dst + k * kNr);
    std::fill_n(dst + k * kNr + cols, kNr - cols, 0.0f);
  }
}

void Tile(std::size_t n, std::size_t it, std::size_t jt, const float* ap,
          const float* bp, float* c, std::size_t ldc) {
  std::size_t i0 = it * kMc, i1 = std::min(i0 + kMc, n);
  std::size_t j0 = jt * kNc, j1 = std::min(j0 + kNc, n);
  for (std::size_t k0 = 0; k0 < n; k0 += kKc) {
    std::size_t kc = std::min(kKc, n - k0);
    for (std::size_t j = j0; j < j1; j += kNr) {
      const float* bk = bp + (j / kNr) * n * kNr + k0 * kNr;
      std::size_t cols = std::min(kNr, n - j);
      for (std::size_t i = i0; i < i1; i += kMr) {
        const float* ak = ap + (i / kMr) * n * kMr + k0 * kMr;
        std::size_t rows = std::min(kMr, n - i);
        float* cij = c + i * ldc + j;
        if (rows == kMr && cols == kNr) {
          Kernel(kc, ak, bk, cij, ldc, k0 > 0);
        } else {
          KernelEdge(kc, ak, bk, cij, ldc, rows, cols, k0 > 0);
        }
      }
    }
  }
}

void Gemm(std::size_t n, const float* a, std::size_t lda, const float* b,
          std::size_t ldb, float* c, std::size_t ldc, bool par) {
  if (n == 0) return;
  std::size_t mp = (n + kMr - 1) / kMr, np = (n + kNr - 1) / kNr;
  std::size_t mt = (n + kMc - 1) / kMc, nt = (n + kNc - 1) / kNc;
  Buffer apack(mp * kMr * n), bpack(np * kNr * n);
  float *ap = apack.data(), *bp = bpack.data();

  // Dynamic scheduling lets the performance cores take more tiles than the
  // efficiency cores.
#pragma omp parallel if (par)
  {
#pragma omp for schedule(static)
    for (std::size_t p = 0; p < mp; ++p) PackA(n, a, lda, p, ap);
#pragma omp for schedule(static)
    for (std::size_t q = 0; q < np; ++q) PackB(n, b, ldb, q, bp);
#pragma omp for schedule(dynamic, 1) collapse(2)
    for (std::size_t it = 0; it < mt; ++it) {
      for (std::size_t jt = 0; jt < nt; ++jt) Tile(n, it, jt, ap, bp, c, ldc);
    }
  }
}

}  // namespace

void GemmNeon(std::size_t n, const float* a, std::size_t lda, const float* b,
              std::size_t ldb, float* c, std::size_t ldc) {
  Gemm(n, a, lda, b, ldb, c, ldc, false);
}

void GemmNeonOmp(std::size_t n, const float* a, std::size_t lda, const float* b,
                 std::size_t ldb, float* c, std::size_t ldc) {
  Gemm(n, a, lda, b, ldb, c, ldc, true);
}

void Simd(std::size_t n, const float* a, const float* b, float* c) {
  GemmNeon(n, a, n, b, n, c, n);
}

void SimdOmp(std::size_t n, const float* a, const float* b, float* c) {
  GemmNeonOmp(n, a, n, b, n, c, n);
}

}  // namespace matmul
