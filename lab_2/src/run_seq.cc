// Sequential reference programs. Usage: see ParseOptions.
#include <string>

#include "loops.h"

int main(int argc, char** argv) {
  std::optional<loops::Options> o = loops::ParseOptions(argc, argv);
  if (!o) return 1;
  const loops::LoopVariant* v = loops::Find(loops::SeqVariants(), o->variant);
  if (!v) {
    std::fprintf(stderr, "unknown variant '%s'; ",
                 std::string(o->variant).c_str());
    loops::PrintVariants(stderr, loops::SeqVariants());
    return 1;
  }

  loops::Grid a(o->ni, o->nj);
  std::optional<loops::Grid> b;
  if (v->uses_b) b.emplace(o->ni, o->nj);
  double min_s = 0, total_s = 0;
  for (int r = 0; r < o->reps; ++r) {
    loops::Init(a);
    if (b) loops::FillZero(*b);
    double t0 = loops::Wtime();
    v->fn(a, b ? &*b : nullptr);
    double dt = loops::Wtime() - t0;
    total_s += dt;
    if (r == 0 || dt < min_s) min_s = dt;
  }

  if (o->header) std::printf("%s\n", loops::kCsvHeader);
  std::printf("%s,seq,%d,%d,1,%d,%.6f,%.6f,%.6f\n",
              std::string(v->name).c_str(), o->ni, o->nj, o->reps, min_s,
              total_s / o->reps, min_s);

  if (o->out && !loops::Write(o->out, b ? *b : a)) return 1;
  return 0;
}
