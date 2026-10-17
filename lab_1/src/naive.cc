#include "matmul.h"

namespace matmul {

void Naive(std::size_t n, const float* a, const float* b, float* c) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (std::size_t k = 0; k < n; ++k) sum += a[i * n + k] * b[k * n + j];
      c[i * n + j] = sum;
    }
  }
}

}  // namespace matmul
