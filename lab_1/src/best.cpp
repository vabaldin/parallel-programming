#include "gemm.hpp"
#include "matmul.hpp"

namespace matmul {

// Below this size starting the threads costs more than they save.
#ifndef BEST_PARALLEL_MIN
#define BEST_PARALLEL_MIN 256
#endif

// One Strassen level saves 1/8 of the multiplications; below this size the
// extra additions cost more than that.
#ifndef BEST_STRASSEN_MIN
#define BEST_STRASSEN_MIN 4096
#endif

void best(std::size_t n, const float *a, const float *b, float *c) {
    if (n < BEST_PARALLEL_MIN)
        gemm_neon(n, a, n, b, n, c, n);
    else if (n < BEST_STRASSEN_MIN)
        gemm_neon_omp(n, a, n, b, n, c, n);
    else
        strassen_mul(n, a, n, b, n, c, n, gemm_neon_omp, n / 2, true);
}

} // namespace matmul
