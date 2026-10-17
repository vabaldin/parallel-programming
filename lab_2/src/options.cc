#include <unistd.h>

#include <cstdint>
#include <cstdlib>

#include "loops.h"

namespace loops {

namespace {

int Positive(const char* s) {
  char* end;
  std::int64_t v = std::strtol(s, &end, 10);
  return (*s && !*end && v > 0 && v <= 1000000) ? static_cast<int>(v) : -1;
}

}  // namespace

std::optional<Options> ParseOptions(int argc, char** argv) {
  Options o;
  bool ok = true;
  int opt;
  while (ok && (opt = getopt(argc, argv, "v:n:m:r:t:o:cH")) != -1) {
    switch (opt) {
      case 'v':
        o.variant = optarg;
        break;
      case 'n':
        o.ni = Positive(optarg);
        break;
      case 'm':
        o.nj = Positive(optarg);
        break;
      case 'r':
        o.reps = Positive(optarg);
        break;
      case 't':
        o.threads = Positive(optarg);
        break;
      case 'o':
        o.out = optarg;
        break;
      case 'c':
        o.check = true;
        break;
      case 'H':
        o.header = true;
        break;
      default:
        ok = false;
        break;
    }
  }
  if (ok && optind == argc && o.ni > 0 && o.nj > 0 && o.reps > 0 &&
      o.threads >= 0) {
    return o;
  }
  std::fprintf(stderr,
               "usage: %s [-v variant] [-n ISIZE] [-m JSIZE] [-r reps]"
               " [-t threads] [-o file] [-c] [-H]\n",
               argv[0]);
  return std::nullopt;
}

}  // namespace loops
