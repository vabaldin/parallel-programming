#include "matmul.h"

namespace matmul {

void Transposed(std::size_t n, const float* a, const float* b, float* c) {
  Buffer scratch(n * n);
  float* bt = scratch.data();
  Transpose(n, b, bt);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (std::size_t k = 0; k < n; ++k) sum += a[i * n + k] * bt[j * n + k];
      c[i * n + j] = sum;
    }
  }
}

}  // namespace matmul
