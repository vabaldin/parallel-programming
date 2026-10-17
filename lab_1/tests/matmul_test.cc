// Usage: test_matmul [variant ...]
#include "matmul.h"

#include <array>
#include <cstdio>
#include <string>

namespace {

constexpr float kRelTol = 1e-3f;
constexpr std::array<std::size_t, 8> kSizes{1, 2, 7, 16, 64, 100, 128, 256};

int Check(const matmul::Variant& v) {
  const matmul::Variant& ref = matmul::Variants().front();
  const std::string name(v.name);
  int failures = 0;

  for (std::size_t n : kSizes) {
    matmul::Matrix a(n), b(n), expected(n), actual(n);

    a.Fill(1.0f);
    b.Fill(1.0f);
    expected.Fill(static_cast<float>(n));
    actual.Fill(-1.0f);
    v(a, b, actual);
    std::size_t bad = matmul::Compare(expected, actual, 0.0f);
    if (bad) {
      std::printf("FAIL %s n=%zu ones: %zu mismatches\n", name.c_str(), n, bad);
      ++failures;
    }

    a.FillRandom(static_cast<unsigned>(2 * n + 1));
    b.FillRandom(static_cast<unsigned>(2 * n + 2));
    ref(a, b, expected);
    actual.Fill(-1.0f);
    v(a, b, actual);
    bad = matmul::Compare(expected, actual, kRelTol);
    if (bad) {
      std::printf("FAIL %s n=%zu random: %zu mismatches\n", name.c_str(), n,
                  bad);
      ++failures;
    }
  }

  std::printf("%s %s\n", failures ? "FAIL" : "ok  ", name.c_str());
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  int failures = 0;
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      const matmul::Variant* v = matmul::Find(argv[i]);
      if (!v) {
        std::printf("unknown variant '%s'\n", argv[i]);
        return 1;
      }
      failures += Check(*v);
    }
  } else {
    for (const matmul::Variant& v : matmul::Variants()) failures += Check(v);
  }
  return failures ? 1 : 0;
}
