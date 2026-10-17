#ifndef LAB_1_INCLUDE_MATMUL_H_
#define LAB_1_INCLUDE_MATMUL_H_

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <span>
#include <string_view>

namespace matmul {

inline constexpr std::size_t kAlign = 64;

// Owning kAlign-aligned array of floats. Throws std::bad_alloc on failure.
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t count);

  float* data() noexcept { return data_.get(); }
  const float* data() const noexcept { return data_.get(); }
  std::size_t size() const noexcept { return size_; }

  float& operator[](std::size_t i) noexcept { return data_[i]; }
  const float& operator[](std::size_t i) const noexcept { return data_[i]; }

 private:
  struct Free {
    void operator()(float* p) const noexcept { std::free(p); }
  };
  std::unique_ptr<float[], Free> data_;
  std::size_t size_ = 0;
};

// Row-major n x n matrix.
class Matrix {
 public:
  explicit Matrix(std::size_t n) : n_(n), buf_(n * n) {}

  std::size_t n() const noexcept { return n_; }
  float* data() noexcept { return buf_.data(); }
  const float* data() const noexcept { return buf_.data(); }

  float& operator()(std::size_t i, std::size_t j) noexcept {
    return buf_[i * n_ + j];
  }
  const float& operator()(std::size_t i, std::size_t j) const noexcept {
    return buf_[i * n_ + j];
  }

  void Fill(float value) noexcept;
  // Values in [0, 1) so that sums do not cancel and relative error stays
  // meaningful.
  void FillRandom(unsigned seed) noexcept;

 private:
  std::size_t n_;
  Buffer buf_;
};

void Transpose(std::size_t n, const float* src, float* dst) noexcept;

// Returns the number of elements that differ by more than
// rel_tol * max(1, |expected|).
std::size_t Compare(const Matrix& expected, const Matrix& actual,
                    float rel_tol) noexcept;

// C = A * B for row-major n x n matrices. Scratch memory is allocated inside
// the variant so that it is included in the timing.
using Fn = void (*)(std::size_t n, const float* a, const float* b, float* c);

struct Variant {
  std::string_view name;
  Fn fn;

  void operator()(const Matrix& a, const Matrix& b, Matrix& c) const {
    fn(a.n(), a.data(), b.data(), c.data());
  }
};

// The first entry is the reference implementation.
std::span<const Variant> Variants() noexcept;
const Variant* Find(std::string_view name) noexcept;

void Naive(std::size_t n, const float* a, const float* b, float* c);
void Transposed(std::size_t n, const float* a, const float* b, float* c);
void Blocked(std::size_t n, const float* a, const float* b, float* c);
void Strassen(std::size_t n, const float* a, const float* b, float* c);
void BlockedOmp(std::size_t n, const float* a, const float* b, float* c);
void Simd(std::size_t n, const float* a, const float* b, float* c);
void SimdOmp(std::size_t n, const float* a, const float* b, float* c);
void Best(std::size_t n, const float* a, const float* b, float* c);

}  // namespace matmul

#endif  // LAB_1_INCLUDE_MATMUL_H_
