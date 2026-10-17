// Usage: test_matmul [variant ...]
#include <array>
#include <cstdio>
#include <string>

#include "matmul.hpp"

namespace {

constexpr float kRelTol = 1e-3f;
constexpr std::array<std::size_t, 10> kSizes{1,   2,   7,   16,  64,
                                             100, 128, 256, 300, 513};

int check(const matmul::Variant &v) {
    const matmul::Variant &ref = matmul::variants().front();
    const std::string name(v.name);
    int failures = 0;

    for (std::size_t n : kSizes) {
        matmul::Matrix a(n), b(n), expected(n), actual(n);

        a.fill(1.0f);
        b.fill(1.0f);
        expected.fill(static_cast<float>(n));
        actual.fill(-1.0f);
        v(a, b, actual);
        std::size_t bad = matmul::compare(expected, actual, 0.0f);
        if (bad) {
            std::printf("FAIL %s n=%zu ones: %zu mismatches\n", name.c_str(),
                        n, bad);
            ++failures;
        }

        a.fill_random(static_cast<unsigned>(2 * n + 1));
        b.fill_random(static_cast<unsigned>(2 * n + 2));
        ref(a, b, expected);
        actual.fill(-1.0f);
        v(a, b, actual);
        bad = matmul::compare(expected, actual, kRelTol);
        if (bad) {
            std::printf("FAIL %s n=%zu random: %zu mismatches\n",
                        name.c_str(), n, bad);
            ++failures;
        }
    }

    std::printf("%s %s\n", failures ? "FAIL" : "ok  ", name.c_str());
    return failures;
}

} // namespace

int main(int argc, char **argv) {
    int failures = 0;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            const matmul::Variant *v = matmul::find(argv[i]);
            if (!v) {
                std::printf("unknown variant '%s'\n", argv[i]);
                return 1;
            }
            failures += check(*v);
        }
    } else {
        for (const matmul::Variant &v : matmul::variants())
            failures += check(v);
    }
    return failures ? 1 : 0;
}
