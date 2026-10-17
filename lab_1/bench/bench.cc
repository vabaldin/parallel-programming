// Usage: bench [-v variant]... [-r reps] [n ...]
// Defaults: all variants, 5 reps, sizes 64 to 4096. gflops uses min_s and
// 2 * n^3 operations.
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "matmul.h"

namespace {

void Run(const matmul::Variant& v, std::size_t n, int reps) {
  matmul::Matrix a(n), b(n), c(n);
  a.FillRandom(1);
  b.FillRandom(2);

  double min_s = 0.0, total_s = 0.0;
  for (int r = 0; r < reps; ++r) {
    double t0 = omp_get_wtime();
    v(a, b, c);
    double dt = omp_get_wtime() - t0;
    total_s += dt;
    if (r == 0 || dt < min_s) min_s = dt;
  }

  double gflops = 2.0 * static_cast<double>(n) * n * n / min_s * 1e-9;
  std::printf("%s,%zu,%d,%d,%.9f,%.9f,%.3f\n", std::string(v.name).c_str(), n,
              omp_get_max_threads(), reps, min_s, total_s / reps, gflops);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<const matmul::Variant*> selected;
  int reps = 5;

  int opt;
  while ((opt = getopt(argc, argv, "v:r:")) != -1) {
    switch (opt) {
      case 'v': {
        const matmul::Variant* v = matmul::Find(optarg);
        if (!v) {
          std::fprintf(stderr, "unknown variant '%s'; available:", optarg);
          for (const matmul::Variant& known : matmul::Variants()) {
            std::fprintf(stderr, " %s", std::string(known.name).c_str());
          }
          std::fprintf(stderr, "\n");
          return 1;
        }
        selected.push_back(v);
        break;
      }
      case 'r':
        reps = std::max(std::atoi(optarg), 1);
        break;
      default:
        std::fprintf(stderr, "usage: %s [-v variant]... [-r reps] [n ...]\n",
                     argv[0]);
        return 1;
    }
  }
  if (selected.empty()) {
    for (const matmul::Variant& v : matmul::Variants()) selected.push_back(&v);
  }

  std::vector<std::size_t> sizes;
  for (int i = optind; i < argc; ++i) {
    sizes.push_back(std::strtoul(argv[i], nullptr, 10));
  }
  if (sizes.empty()) sizes = {64, 128, 256, 512, 1024, 2048, 4096};

  std::printf("variant,n,threads,reps,min_s,mean_s,gflops\n");
  for (std::size_t n : sizes) {
    for (const matmul::Variant* v : selected) Run(*v, n, reps);
  }
  return 0;
}
