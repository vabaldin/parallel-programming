#include <algorithm>
#include <array>

#include "matmul.hpp"

namespace matmul {

namespace {

constexpr std::array kVariants{
    Variant{"naive", naive},
    Variant{"transposed", transposed},
    Variant{"blocked", blocked},
    Variant{"strassen", strassen},
    Variant{"blocked_omp", blocked_omp},
    Variant{"simd", simd},
    Variant{"simd_omp", simd_omp},
    Variant{"best", best},
};

} // namespace

std::span<const Variant> variants() noexcept { return kVariants; }

const Variant *find(std::string_view name) noexcept {
    auto it = std::find_if(kVariants.begin(), kVariants.end(),
                           [name](const Variant &v) { return v.name == name; });
    return it == kVariants.end() ? nullptr : &*it;
}

} // namespace matmul
