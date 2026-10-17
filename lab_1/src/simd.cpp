#include <algorithm>

#include <arm_neon.h>

#include "gemm.hpp"
#include "matmul.hpp"

namespace matmul {

namespace {

// The 8 x 12 micro-tile of C lives in 24 of the 32 NEON registers. A KC x NR
// panel of B (24 KiB) stays in L1 while the MC x KC block of A (128 KiB)
// comes from L2. MC and NC are multiples of MR and NR, and an MC x NC tile
// of C is one unit of parallel work. The values were tuned on the M3 Pro.
#ifndef KC
#define KC 512
#endif
#ifndef MC
#define MC 64
#endif
#ifndef NC
#define NC 192
#endif
constexpr std::size_t kMr = 8;
constexpr std::size_t kNr = 12;
constexpr std::size_t kKc = KC;
constexpr std::size_t kMc = MC;
constexpr std::size_t kNc = NC;

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

// C[8 x 12] (+)= A panel * B panel; a holds kMr floats per k, b holds kNr.
void kernel(std::size_t kc, const float *__restrict a,
            const float *__restrict b, float *__restrict c, std::size_t ldc,
            bool accumulate) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    LOAD_ROW(0) LOAD_ROW(1) LOAD_ROW(2) LOAD_ROW(3)
    LOAD_ROW(4) LOAD_ROW(5) LOAD_ROW(6) LOAD_ROW(7)

    for (std::size_t k = 0; k < kc; ++k) {
        float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4),
                    b2 = vld1q_f32(b + 8);
        FMA_ROW(0, a0, 0) FMA_ROW(1, a0, 1) FMA_ROW(2, a0, 2)
        FMA_ROW(3, a0, 3) FMA_ROW(4, a1, 0) FMA_ROW(5, a1, 1)
        FMA_ROW(6, a1, 2) FMA_ROW(7, a1, 3)
        a += kMr;
        b += kNr;
    }

    STORE_ROW(0) STORE_ROW(1) STORE_ROW(2) STORE_ROW(3)
    STORE_ROW(4) STORE_ROW(5) STORE_ROW(6) STORE_ROW(7)
}

#undef LOAD_ROW
#undef FMA_ROW
#undef STORE_ROW

// Partial tiles at the right and bottom edges go through a full-size buffer.
void kernel_edge(std::size_t kc, const float *a, const float *b, float *c,
                 std::size_t ldc, std::size_t rows, std::size_t cols,
                 bool accumulate) {
    float buf[kMr * kNr];
    if (accumulate)
        for (std::size_t i = 0; i < rows; ++i)
            std::copy_n(c + i * ldc, cols, buf + i * kNr);
    kernel(kc, a, b, buf, kNr, accumulate);
    for (std::size_t i = 0; i < rows; ++i)
        std::copy_n(buf + i * kNr, cols, c + i * ldc);
}

// Row panel p of A: element (k, r) at p * n * kMr + k * kMr + r. Rows past n
// are zero.
void pack_a(std::size_t n, const float *a, std::size_t lda, std::size_t p,
            float *dst) {
    std::size_t i0 = p * kMr, rows = std::min(kMr, n - i0);
    dst += p * n * kMr;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t r = 0;
        for (; r < rows; ++r)
            dst[k * kMr + r] = a[(i0 + r) * lda + k];
        for (; r < kMr; ++r)
            dst[k * kMr + r] = 0.0f;
    }
}

// Column panel q of B: element (k, j) at q * n * kNr + k * kNr + j.
void pack_b(std::size_t n, const float *b, std::size_t ldb, std::size_t q,
            float *dst) {
    std::size_t j0 = q * kNr, cols = std::min(kNr, n - j0);
    dst += q * n * kNr;
    for (std::size_t k = 0; k < n; ++k) {
        std::copy_n(b + k * ldb + j0, cols, dst + k * kNr);
        std::fill_n(dst + k * kNr + cols, kNr - cols, 0.0f);
    }
}

void tile(std::size_t n, std::size_t it, std::size_t jt, const float *ap,
          const float *bp, float *c, std::size_t ldc) {
    std::size_t i0 = it * kMc, i1 = std::min(i0 + kMc, n);
    std::size_t j0 = jt * kNc, j1 = std::min(j0 + kNc, n);
    for (std::size_t k0 = 0; k0 < n; k0 += kKc) {
        std::size_t kc = std::min(kKc, n - k0);
        for (std::size_t j = j0; j < j1; j += kNr) {
            const float *bk = bp + (j / kNr) * n * kNr + k0 * kNr;
            std::size_t cols = std::min(kNr, n - j);
            for (std::size_t i = i0; i < i1; i += kMr) {
                const float *ak = ap + (i / kMr) * n * kMr + k0 * kMr;
                std::size_t rows = std::min(kMr, n - i);
                float *cij = c + i * ldc + j;
                if (rows == kMr && cols == kNr)
                    kernel(kc, ak, bk, cij, ldc, k0 > 0);
                else
                    kernel_edge(kc, ak, bk, cij, ldc, rows, cols, k0 > 0);
            }
        }
    }
}

void gemm(std::size_t n, const float *a, std::size_t lda, const float *b,
          std::size_t ldb, float *c, std::size_t ldc, bool par) {
    if (n == 0)
        return;
    std::size_t mp = (n + kMr - 1) / kMr, np = (n + kNr - 1) / kNr;
    std::size_t mt = (n + kMc - 1) / kMc, nt = (n + kNc - 1) / kNc;
    Buffer apack(mp * kMr * n), bpack(np * kNr * n);
    float *ap = apack.data(), *bp = bpack.data();

    // Dynamic scheduling lets the performance cores take more tiles than the
    // efficiency cores.
#pragma omp parallel if (par)
    {
#pragma omp for schedule(static)
        for (std::size_t p = 0; p < mp; ++p)
            pack_a(n, a, lda, p, ap);
#pragma omp for schedule(static)
        for (std::size_t q = 0; q < np; ++q)
            pack_b(n, b, ldb, q, bp);
#pragma omp for schedule(dynamic, 1) collapse(2)
        for (std::size_t it = 0; it < mt; ++it)
            for (std::size_t jt = 0; jt < nt; ++jt)
                tile(n, it, jt, ap, bp, c, ldc);
    }
}

} // namespace

void gemm_neon(std::size_t n, const float *a, std::size_t lda, const float *b,
               std::size_t ldb, float *c, std::size_t ldc) {
    gemm(n, a, lda, b, ldb, c, ldc, false);
}

void gemm_neon_omp(std::size_t n, const float *a, std::size_t lda,
                   const float *b, std::size_t ldb, float *c,
                   std::size_t ldc) {
    gemm(n, a, lda, b, ldb, c, ldc, true);
}

void simd(std::size_t n, const float *a, const float *b, float *c) {
    gemm_neon(n, a, n, b, n, c, n);
}

void simd_omp(std::size_t n, const float *a, const float *b, float *c) {
    gemm_neon_omp(n, a, n, b, n, c, n);
}

} // namespace matmul
