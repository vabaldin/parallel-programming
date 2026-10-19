#ifndef LAB_2_INCLUDE_LOOPS_H_
#define LAB_2_INCLUDE_LOOPS_H_

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace loops {

// Row-major ni x nj array of doubles, 64-byte aligned. The memory is left
// uninitialized so that an MPI rank maps only the pages it touches. Throws
// std::bad_alloc on failure.
class Grid {
 public:
  Grid(int ni, int nj);

  int ni() const noexcept { return ni_; }
  int nj() const noexcept { return nj_; }
  std::size_t size() const noexcept {
    return static_cast<std::size_t>(ni_) * static_cast<std::size_t>(nj_);
  }

  double* data() noexcept { return data_.get(); }
  const double* data() const noexcept { return data_.get(); }
  double* Row(int i) noexcept {
    return data() + static_cast<std::size_t>(i) * nj_;
  }
  const double* Row(int i) const noexcept {
    return data() + static_cast<std::size_t>(i) * nj_;
  }

 private:
  struct Free {
    void operator()(double* p) const noexcept { std::free(p); }
  };
  int ni_, nj_;
  std::unique_ptr<double[], Free> data_;
};

// a[i][j] = 10 * i + j for the whole grid, or for a[i][jlo..jhi) only.
void Init(Grid& a) noexcept;
void InitSpan(Grid& a, int i, int jlo, int jhi) noexcept;
void FillZero(Grid& a) noexcept;

// Writes the grid in the manual's format: "%f " per element, one row per
// line. Returns false and prints the reason on failure.
bool Write(const char* path, const Grid& x);

// Number of elements whose bit patterns differ.
std::size_t Diff(const Grid& x, const Grid& y) noexcept;

double Wtime() noexcept;

// The loop bodies are shared so that every implementation performs exactly
// the same floating-point operations and the outputs match bit for bit.
inline void RefRow(double* a_i, int nj) {
  for (int j = 0; j < nj; ++j) a_i[j] = std::sin(2 * a_i[j]);
}

// 1b: a[i][j] = sin(5 * a[i-2][j+3]) for j in [jlo, jhi).
inline void V1bRow(double* a_i, const double* a_im2, int jlo, int jhi) {
  for (int j = jlo; j < jhi; ++j) a_i[j] = std::sin(5 * a_im2[j + 3]);
}

// 2b: a[i][j] = sin(0.2 * a[i+1][j-6]) for j in [jlo, jhi).
inline void V2bRow(double* a_i, const double* a_ip1, int jlo, int jhi) {
  for (int j = jlo; j < jhi; ++j) a_i[j] = std::sin(0.2 * a_ip1[j - 6]);
}

// 3b, first loop: a[i][j] = sin(0.01 * a[i][j]).
inline void V3bRowA(double* a_i, int nj) {
  for (int j = 0; j < nj; ++j) a_i[j] = std::sin(0.01 * a_i[j]);
}

// 3b, second loop: b[i][j] = a[i+1][j-3] * 2.
inline void V3bRowB(double* b_i, const double* a_ip1, int nj) {
  for (int j = 3; j < nj; ++j) b_i[j] = a_ip1[j - 3] * 2;
}

// A loop nest from the manual. b is null for variants that do not use it.
using LoopFn = void (*)(Grid& a, Grid* b);

struct LoopVariant {
  std::string_view name;
  bool uses_b;
  LoopFn fn;
};

// Sequential reference implementations: ref, 1b, 2b, 3b.
std::span<const LoopVariant> SeqVariants() noexcept;
const LoopVariant* Find(std::span<const LoopVariant> variants,
                        std::string_view name) noexcept;
void PrintVariants(std::FILE* f, std::span<const LoopVariant> variants);

void SeqRef(Grid& a, Grid* b);
void Seq1b(Grid& a, Grid* b);
void Seq2b(Grid& a, Grid* b);
void Seq3b(Grid& a, Grid* b);

struct Options {
  std::string_view variant = "ref";
  int ni = 5000, nj = 5000;
  int reps = 3;
  int threads = 0;
  const char* out = nullptr;
  bool check = false;
  bool header = false;
};

// Parses -v variant -n ISIZE -m JSIZE -r reps -t threads -o file -c -H.
// Prints usage and returns nothing on error.
std::optional<Options> ParseOptions(int argc, char** argv);

inline constexpr const char* kCsvHeader =
    "variant,impl,isize,jsize,workers,reps,min_s,mean_s,comp_s";

// Runs the sequential version of variant `name` on a fresh grid and compares
// its result (a, or b for variants that use b) with `result`. Prints the
// outcome to stderr and returns the number of mismatching elements.
std::size_t CheckAgainstSeq(std::string_view name, const Grid& result);

}  // namespace loops

#endif  // LAB_2_INCLUDE_LOOPS_H_
