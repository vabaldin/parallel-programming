#include "gemm.h"
#include "matmul.h"

namespace matmul {

// One Strassen level saves 1/8 of the multiplications; below this size the
// extra additions cost more than that.
constexpr std::size_t kBestStrassenMin = 4096;

void Best(std::size_t n, const float* a, const float* b, float* c) {
  if (n >= kBestStrassenMin) {
    StrassenMul(n, a, n, b, n, c, n, GemmNeonOmp, n / 2, true);
  } else {
    GemmNeonOmp(n, a, n, b, n, c, n);
  }
}

}  // namespace matmul
