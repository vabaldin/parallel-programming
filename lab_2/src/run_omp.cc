// OpenMP versions of ref, 2b and 3b. Usage: see ParseOptions; -t sets the
// thread count (default OMP_NUM_THREADS).
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "loops.h"

namespace {

using loops::Grid;

// The cores are not equal (5 performance, 6 efficiency), so rows are handed
// out dynamically in small chunks instead of in equal static blocks.
constexpr int kRowChunk = 8;
constexpr int kBlocksPerThread = 8;

void ParRef(Grid& a, Grid*) {
  int ni = a.ni(), nj = a.nj();
#pragma omp parallel for schedule(dynamic, kRowChunk)
  for (int i = 0; i < ni; ++i) loops::RefRow(a.Row(i), nj);
}

int BlockLo(int rows, int nb, int k) {
  return static_cast<int>(static_cast<std::int64_t>(rows) * k / nb);
}

// Anti-dependence (i, j) -> (i+1, j-6): row i must read row i+1 before row
// i+1 is overwritten. The rows are cut into blocks that run in any order,
// each in sequential order inside. The only row a block reads but does not
// own is the first row of the next block, so copies of those rows are taken
// before any block starts (the barrier of the first omp for).
void Par2b(Grid& a, Grid*) {
  int nj = a.nj(), rows = a.ni() - 1;
  if (rows <= 0 || nj <= 6) return;
  int nb = std::min(kBlocksPerThread * omp_get_max_threads(), rows);
  std::size_t w = static_cast<std::size_t>(nj - 6);
  auto halo = std::make_unique_for_overwrite<double[]>(nb * w);
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int k = 0; k < nb; ++k) {
      std::copy_n(a.Row(BlockLo(rows, nb, k + 1)), w, halo.get() + k * w);
    }
#pragma omp for schedule(dynamic, 1)
    for (int k = 0; k < nb; ++k) {
      int lo = BlockLo(rows, nb, k), hi = BlockLo(rows, nb, k + 1);
      for (int i = lo; i < hi - 1; ++i) {
        loops::V2bRow(a.Row(i), a.Row(i + 1), 6, nj);
      }
      loops::V2bRow(a.Row(hi - 1), halo.get() + k * w, 6, nj);
    }
  }
}

// Alternative for 2b: the inner loop has no dependence, so split j and
// keep i sequential. The barrier at the end of each omp for orders the reads
// of row i+1 before its writes.
void Par2bCols(Grid& a, Grid*) {
  int ni = a.ni(), nj = a.nj();
#pragma omp parallel
  for (int i = 0; i < ni - 1; ++i) {
    double* a_i = a.Row(i);
    const double* a_ip1 = a.Row(i + 1);
#pragma omp for schedule(static)
    for (int j = 6; j < nj; ++j) loops::V2bRow(a_i, a_ip1, j, j + 1);
  }
}

// Both nests are free of loop-carried dependences. The second one reads
// a[i+1] produced by the first, so the barrier between them is required.
void Par3b(Grid& a, Grid* b) {
  int ni = a.ni(), nj = a.nj();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic, kRowChunk)
    for (int i = 0; i < ni; ++i) loops::V3bRowA(a.Row(i), nj);
#pragma omp for schedule(static)
    for (int i = 0; i < ni - 1; ++i) {
      loops::V3bRowB(b->Row(i), a.Row(i + 1), nj);
    }
  }
}

constexpr std::array kVariants{
    loops::LoopVariant{"ref", false, ParRef},
    loops::LoopVariant{"2b", false, Par2b},
    loops::LoopVariant{"2b-cols", false, Par2bCols},
    loops::LoopVariant{"3b", true, Par3b},
};

// Sequential variant that each OpenMP variant is checked against.
std::string_view SeqName(std::string_view name) {
  return name == "2b-cols" ? "2b" : name;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<loops::Options> o = loops::ParseOptions(argc, argv);
  if (!o) return 1;
  const loops::LoopVariant* v = loops::Find(kVariants, o->variant);
  if (!v) {
    std::fprintf(stderr, "unknown variant '%s'; ",
                 std::string(o->variant).c_str());
    loops::PrintVariants(stderr, kVariants);
    return 1;
  }
  if (o->threads > 0) omp_set_num_threads(o->threads);

  Grid a(o->ni, o->nj);
  std::optional<Grid> b;
  if (v->uses_b) b.emplace(o->ni, o->nj);
  double min_s = 0, total_s = 0;
  for (int r = 0; r < o->reps; ++r) {
    loops::Init(a);
    if (b) loops::FillZero(*b);
    double t0 = omp_get_wtime();
    v->fn(a, b ? &*b : nullptr);
    double dt = omp_get_wtime() - t0;
    total_s += dt;
    if (r == 0 || dt < min_s) min_s = dt;
  }

  if (o->header) std::printf("%s\n", loops::kCsvHeader);
  std::printf("%s,omp,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
              std::string(v->name).c_str(), o->ni, o->nj, omp_get_max_threads(),
              o->reps, min_s, total_s / o->reps, min_s);

  const Grid& result = b ? *b : a;
  int status = 0;
  if (o->out && !loops::Write(o->out, result)) status = 1;
  if (o->check && loops::CheckAgainstSeq(SeqName(v->name), result) != 0) {
    status = 1;
  }
  return status;
}
