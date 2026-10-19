#include "gemm.h"
#include "matmul.h"

namespace matmul {

// Below this size starting the threads costs more than they save.
constexpr std::size_t kBestParallelMin = 256;

// One Strassen level saves 1/8 of the multiplications; below this size the
// extra additions cost more than that.
constexpr std::size_t kBestStrassenMin = 4096;

void Best(std::size_t n, const float* a, const float* b, float* c) {
  if (n < kBestParallelMin) {
    GemmNeon(n, a, n, b, n, c, n);
  } else if (n < kBestStrassenMin) {
    GemmNeonOmp(n, a, n, b, n, c, n);
  } else {
    StrassenMul(n, a, n, b, n, c, n, GemmNeonOmp, n / 2, true);
  }
}

}  // namespace matmul
