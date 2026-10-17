#include <algorithm>
#include <array>

#include "matmul.h"

namespace matmul {

namespace {

constexpr std::array kVariants{
    Variant{"naive", Naive},
    Variant{"transposed", Transposed},
    Variant{"blocked", Blocked},
    Variant{"strassen", Strassen},
};

}  // namespace

std::span<const Variant> Variants() noexcept { return kVariants; }

const Variant* Find(std::string_view name) noexcept {
  auto it = std::find_if(kVariants.begin(), kVariants.end(),
                         [name](const Variant& v) { return v.name == name; });
  return it == kVariants.end() ? nullptr : &*it;
}

}  // namespace matmul
