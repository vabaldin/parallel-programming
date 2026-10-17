#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <new>
#include <string>

#include "loops.h"

namespace loops {

namespace {

constexpr std::size_t kAlign = 64;

struct Close {
  void operator()(std::FILE* f) const noexcept { std::fclose(f); }
};

}  // namespace

Grid::Grid(int ni, int nj) : ni_(ni), nj_(nj) {
  std::size_t bytes = size() * sizeof(double);
  bytes = std::max((bytes + kAlign - 1) / kAlign * kAlign, kAlign);
  data_.reset(static_cast<double*>(std::aligned_alloc(kAlign, bytes)));
  if (!data_) throw std::bad_alloc();
}

void InitSpan(Grid& a, int i, int jlo, int jhi) noexcept {
  double* a_i = a.Row(i);
  for (int j = jlo; j < jhi; ++j) a_i[j] = 10 * i + j;
}

void Init(Grid& a) noexcept {
  for (int i = 0; i < a.ni(); ++i) InitSpan(a, i, 0, a.nj());
}

void FillZero(Grid& a) noexcept { std::fill_n(a.data(), a.size(), 0.0); }

bool Write(const char* path, const Grid& x) {
  std::unique_ptr<std::FILE, Close> f(std::fopen(path, "w"));
  if (!f) {
    std::perror(path);
    return false;
  }
  for (int i = 0; i < x.ni(); ++i) {
    const double* x_i = x.Row(i);
    for (int j = 0; j < x.nj(); ++j) std::fprintf(f.get(), "%f ", x_i[j]);
    std::fprintf(f.get(), "\n");
  }
  // Close explicitly: a failed flush is only reported by fclose.
  if (std::fclose(f.release()) != 0) {
    std::perror(path);
    return false;
  }
  return true;
}

std::size_t Diff(const Grid& x, const Grid& y) noexcept {
  const double *px = x.data(), *py = y.data();
  std::size_t n = x.size(), count = 0;
  for (std::size_t k = 0; k < n; ++k) {
    count += std::bit_cast<std::uint64_t>(px[k]) !=
             std::bit_cast<std::uint64_t>(py[k]);
  }
  return count;
}

double Wtime() noexcept {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::size_t CheckAgainstSeq(std::string_view name, const Grid& result) {
  const std::string sname(name);
  const LoopVariant* v = Find(SeqVariants(), name);
  if (!v) {
    std::fprintf(stderr, "check: no sequential variant '%s'\n", sname.c_str());
    return 1;
  }
  int ni = result.ni(), nj = result.nj();
  Grid a(ni, nj);
  std::optional<Grid> b;
  Init(a);
  if (v->uses_b) {
    b.emplace(ni, nj);
    FillZero(*b);
  }
  v->fn(a, b ? &*b : nullptr);
  std::size_t bad = Diff(b ? *b : a, result);
  if (bad) {
    std::fprintf(stderr, "check FAIL %s %dx%d: %zu of %zu elements differ\n",
                 sname.c_str(), ni, nj, bad, result.size());
  } else {
    std::fprintf(stderr, "check ok %s %dx%d\n", sname.c_str(), ni, nj);
  }
  return bad;
}

}  // namespace loops
