#include <algorithm>

#include "gemm.hpp"
#include "matmul.hpp"

namespace matmul {

namespace {

// A cutoff of 32 is about 4% faster but more than doubles the rounding error
// at n = 4096.
#ifndef STRASSEN_CUTOFF
#define STRASSEN_CUTOFF 64
#endif

// z = x + sign * y
void add(std::size_t h, const float *x, std::size_t ldx, const float *y,
         std::size_t ldy, float sign, float *z, std::size_t ldz, bool par) {
#pragma omp parallel for schedule(static) if (par)
    for (std::size_t i = 0; i < h; ++i)
        for (std::size_t j = 0; j < h; ++j)
            z[i * ldz + j] = x[i * ldx + j] + sign * y[i * ldy + j];
}

void put(std::size_t h, const float *m, float *c, std::size_t ldc, bool par) {
#pragma omp parallel for schedule(static) if (par)
    for (std::size_t i = 0; i < h; ++i)
        std::copy_n(m + i * h, h, c + i * ldc);
}

// c += sign * m
void acc(std::size_t h, const float *m, float sign, float *c, std::size_t ldc,
         bool par) {
#pragma omp parallel for schedule(static) if (par)
    for (std::size_t i = 0; i < h; ++i)
        for (std::size_t j = 0; j < h; ++j)
            c[i * ldc + j] += sign * m[i * h + j];
}

std::size_t workspace_size(std::size_t n, std::size_t cutoff) {
    std::size_t total = 0;
    for (; n > cutoff; n /= 2)
        total += 3 * (n / 2) * (n / 2);
    return total;
}

// Every size above the cutoff is even here; strassen_mul pads to make it
// so. The seven products are accumulated straight into the quadrants of C,
// so one level needs only three h x h temporaries.
void rec(std::size_t n, const float *a, std::size_t lda, const float *b,
         std::size_t ldb, float *c, std::size_t ldc, GemmFn base,
         std::size_t cutoff, float *ws, bool par) {
    if (n <= cutoff) {
        base(n, a, lda, b, ldb, c, ldc);
        return;
    }

    std::size_t h = n / 2;
    float *t1 = ws, *t2 = ws + h * h, *m = ws + 2 * h * h;
    float *next = ws + 3 * h * h;
    const float *a11 = a, *a12 = a + h, *a21 = a + h * lda,
                *a22 = a + h * lda + h;
    const float *b11 = b, *b12 = b + h, *b21 = b + h * ldb,
                *b22 = b + h * ldb + h;
    float *c11 = c, *c12 = c + h, *c21 = c + h * ldc, *c22 = c + h * ldc + h;

    add(h, a11, lda, a22, lda, 1.0f, t1, h, par);
    add(h, b11, ldb, b22, ldb, 1.0f, t2, h, par);
    rec(h, t1, h, t2, h, m, h, base, cutoff, next, par);
    put(h, m, c11, ldc, par);
    put(h, m, c22, ldc, par);

    add(h, a21, lda, a22, lda, 1.0f, t1, h, par);
    rec(h, t1, h, b11, ldb, m, h, base, cutoff, next, par);
    put(h, m, c21, ldc, par);
    acc(h, m, -1.0f, c22, ldc, par);

    add(h, b12, ldb, b22, ldb, -1.0f, t2, h, par);
    rec(h, a11, lda, t2, h, m, h, base, cutoff, next, par);
    put(h, m, c12, ldc, par);
    acc(h, m, 1.0f, c22, ldc, par);

    add(h, b21, ldb, b11, ldb, -1.0f, t2, h, par);
    rec(h, a22, lda, t2, h, m, h, base, cutoff, next, par);
    acc(h, m, 1.0f, c11, ldc, par);
    acc(h, m, 1.0f, c21, ldc, par);

    add(h, a11, lda, a12, lda, 1.0f, t1, h, par);
    rec(h, t1, h, b22, ldb, m, h, base, cutoff, next, par);
    acc(h, m, -1.0f, c11, ldc, par);
    acc(h, m, 1.0f, c12, ldc, par);

    add(h, a21, lda, a11, lda, -1.0f, t1, h, par);
    add(h, b11, ldb, b12, ldb, 1.0f, t2, h, par);
    rec(h, t1, h, t2, h, m, h, base, cutoff, next, par);
    acc(h, m, 1.0f, c22, ldc, par);

    add(h, a12, lda, a22, lda, -1.0f, t1, h, par);
    add(h, b21, ldb, b22, ldb, 1.0f, t2, h, par);
    rec(h, t1, h, t2, h, m, h, base, cutoff, next, par);
    acc(h, m, 1.0f, c11, ldc, par);
}

Buffer pad_copy(std::size_t n, const float *src, std::size_t lds,
                std::size_t m) {
    Buffer dst(m * m);
    std::fill_n(dst.data(), m * m, 0.0f);
    for (std::size_t i = 0; i < n; ++i)
        std::copy_n(src + i * lds, n, dst.data() + i * m);
    return dst;
}

} // namespace

void strassen_mul(std::size_t n, const float *a, std::size_t lda,
                  const float *b, std::size_t ldb, float *c, std::size_t ldc,
                  GemmFn base, std::size_t cutoff, bool parallel) {
    // Pad n up to leaf * 2^levels with leaf <= cutoff.
    std::size_t levels = 0, leaf = n;
    while (leaf > cutoff) {
        ++levels;
        leaf = (n + (std::size_t{1} << levels) - 1) >> levels;
    }
    std::size_t m = leaf << levels;

    Buffer ws(workspace_size(m, cutoff));
    if (m == n) {
        rec(n, a, lda, b, ldb, c, ldc, base, cutoff, ws.data(), parallel);
        return;
    }
    Buffer ap = pad_copy(n, a, lda, m), bp = pad_copy(n, b, ldb, m);
    Buffer cp(m * m);
    rec(m, ap.data(), m, bp.data(), m, cp.data(), m, base, cutoff, ws.data(),
        parallel);
    for (std::size_t i = 0; i < n; ++i)
        std::copy_n(cp.data() + i * m, n, c + i * ldc);
}

void strassen(std::size_t n, const float *a, const float *b, float *c) {
    strassen_mul(n, a, n, b, n, c, n, gemm_blocked, STRASSEN_CUTOFF, false);
}

} // namespace matmul
