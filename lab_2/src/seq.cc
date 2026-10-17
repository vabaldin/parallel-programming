#include <algorithm>
#include <array>
#include <string>

#include "loops.h"

namespace loops {

void SeqRef(Grid& a, Grid*) {
  for (int i = 0; i < a.ni(); ++i) RefRow(a.Row(i), a.nj());
}

void Seq1b(Grid& a, Grid*) {
  for (int i = 2; i < a.ni(); ++i) {
    V1bRow(a.Row(i), a.Row(i - 2), 0, a.nj() - 3);
  }
}

void Seq2b(Grid& a, Grid*) {
  for (int i = 0; i < a.ni() - 1; ++i) {
    V2bRow(a.Row(i), a.Row(i + 1), 6, a.nj());
  }
}

void Seq3b(Grid& a, Grid* b) {
  for (int i = 0; i < a.ni(); ++i) V3bRowA(a.Row(i), a.nj());
  for (int i = 0; i < a.ni() - 1; ++i) V3bRowB(b->Row(i), a.Row(i + 1), a.nj());
}

namespace {

constexpr std::array kSeqVariants{
    LoopVariant{"ref", false, SeqRef},
    LoopVariant{"1b", false, Seq1b},
    LoopVariant{"2b", false, Seq2b},
    LoopVariant{"3b", true, Seq3b},
};

}  // namespace

std::span<const LoopVariant> SeqVariants() noexcept { return kSeqVariants; }

const LoopVariant* Find(std::span<const LoopVariant> variants,
                        std::string_view name) noexcept {
  auto it =
      std::find_if(variants.begin(), variants.end(),
                   [name](const LoopVariant& v) { return v.name == name; });
  return it == variants.end() ? nullptr : &*it;
}

void PrintVariants(std::FILE* f, std::span<const LoopVariant> variants) {
  std::fprintf(f, "available:");
  for (const LoopVariant& v : variants) {
    std::fprintf(f, " %s", std::string(v.name).c_str());
  }
  std::fprintf(f, "\n");
}

}  // namespace loops
