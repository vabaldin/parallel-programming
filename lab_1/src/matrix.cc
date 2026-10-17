#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>

#include "matmul.h"

namespace matmul {

Buffer::Buffer(std::size_t count) : size_(count) {
  std::size_t bytes = count * sizeof(float);
  bytes = std::max((bytes + kAlign - 1) / kAlign * kAlign, kAlign);
  data_.reset(static_cast<float*>(std::aligned_alloc(kAlign, bytes)));
  if (!data_) throw std::bad_alloc();
}

void Matrix::Fill(float value) noexcept { std::fill_n(data(), n_ * n_, value); }

void Matrix::FillRandom(unsigned seed) noexcept {
  std::uint32_t state = seed * 2654435761u + 1u;
  float* m = data();
  for (std::size_t i = 0; i < n_ * n_; ++i) {
    state = state * 1664525u + 1013904223u;
    m[i] = static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
  }
}

void Transpose(std::size_t n, const float* src, float* dst) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) dst[j * n + i] = src[i * n + j];
  }
}

std::size_t Compare(const Matrix& expected, const Matrix& actual,
                    float rel_tol) noexcept {
  const float *e = expected.data(), *a = actual.data();
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < expected.n() * expected.n(); ++i) {
    float diff = std::fabs(e[i] - a[i]);
    if (!(diff <= rel_tol * std::max(1.0f, std::fabs(e[i])))) ++mismatches;
  }
  return mismatches;
}

}  // namespace matmul
