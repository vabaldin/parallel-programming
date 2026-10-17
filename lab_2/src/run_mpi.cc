// MPI versions of ref and 1b. Run with mpirun -np P. Every rank allocates
// the whole grid but initializes and touches only the part it computes, so
// the other pages are never mapped; rank 0 initializes everything, collects
// the result, prints the CSV line and writes or checks the output. min_s
// covers computation plus collection on rank 0, comp_s only the computation
// (both are the maximum over ranks).
#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "loops.h"

namespace {

using loops::Grid;

class Session {
 public:
  Session(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
  ~Session() { MPI_Finalize(); }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
};

class Datatype {
 public:
  Datatype() = default;
  explicit Datatype(MPI_Datatype t) : t_(t) { MPI_Type_commit(&t_); }
  ~Datatype() {
    if (t_ != MPI_DATATYPE_NULL) MPI_Type_free(&t_);
  }
  Datatype(Datatype&& other) noexcept
      : t_(std::exchange(other.t_, MPI_DATATYPE_NULL)) {}
  Datatype& operator=(Datatype&& other) noexcept {
    std::swap(t_, other.t_);
    return *this;
  }

  MPI_Datatype get() const noexcept { return t_; }

 private:
  MPI_Datatype t_ = MPI_DATATYPE_NULL;
};

int rank, nprocs;

int BlockLo(int n, int p) {
  return static_cast<int>(static_cast<std::int64_t>(n) * p / nprocs);
}

void InitRef(Grid& a) {
  for (int i = BlockLo(a.ni(), rank); i < BlockLo(a.ni(), rank + 1); ++i) {
    loops::InitSpan(a, i, 0, a.nj());
  }
}

// ref: no dependences, contiguous blocks of rows, MPI_Gatherv to rank 0.
void MpiRef(Grid& a, double& comp_s) {
  int ni = a.ni(), nj = a.nj();
  double t0 = MPI_Wtime();
  int lo = BlockLo(ni, rank), hi = BlockLo(ni, rank + 1);
  for (int i = lo; i < hi; ++i) loops::RefRow(a.Row(i), nj);
  comp_s = MPI_Wtime() - t0;

  if (nprocs == 1) return;
  std::vector<int> counts(nprocs), displs(nprocs);
  for (int p = 0; p < nprocs; ++p) {
    counts[p] = (BlockLo(ni, p + 1) - BlockLo(ni, p)) * nj;
    displs[p] = BlockLo(ni, p) * nj;
  }
  if (rank == 0) {
    MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, a.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(a.Row(lo), counts[rank], MPI_DOUBLE, nullptr, nullptr, nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

// 1b: a[i][j] = sin(5 * a[i-2][j+3]), distance vector (2, -3).
// With the skew k = j + 3 * (i / 2), iteration (i, j) and its source
// (i-2, j+3) have the same k, so every k is an independent chain and the
// k axis can be split between ranks with no communication during the
// computation. In each row a rank owns one contiguous run of columns.
int Skew(int i) { return 3 * (i / 2); }

// Splits the k range so that every rank gets about the same number of
// computed elements. The result has nprocs + 1 entries.
std::vector<int> Split1b(int ni, int nj) {
  int nk = nj - 4 + Skew(ni - 1) + 2;
  std::vector<std::int64_t> cnt(nk);
  for (int i = 2; i < ni; ++i) {
    cnt[Skew(i)] += 1;
    cnt[nj - 3 + Skew(i)] -= 1;
  }
  std::int64_t total = 0, prefix = 0;
  for (int k = 0; k < nk; ++k) {
    prefix += cnt[k];
    cnt[k] = prefix;
    total += prefix;
  }
  std::vector<int> bounds(nprocs + 1);
  std::int64_t acc = 0;
  int k = 0;
  for (int p = 1; p < nprocs; ++p) {
    std::int64_t target = total * p / nprocs;
    while (k < nk && acc < target) acc += cnt[k++];
    bounds[p] = k;
  }
  bounds[nprocs] = nk;
  return bounds;
}

// Columns of row i in the chains of rank p, clipped to [0, jmax).
std::pair<int, int> Cols1b(int i, int jmax, const std::vector<int>& bounds,
                           int p) {
  return {std::max(bounds[p] - Skew(i), 0),
          std::min(bounds[p + 1] - Skew(i), jmax)};
}

// Every element of the rank's chains, including the initial values at their
// ends in rows 0 and 1 and in the last three columns.
void Init1b(Grid& a) {
  int ni = a.ni(), nj = a.nj();
  if (ni <= 2 || nj <= 3) return;
  std::vector<int> bounds = Split1b(ni, nj);
  for (int i = 0; i < ni; ++i) {
    auto [jlo, jhi] = Cols1b(i, nj, bounds, rank);
    loops::InitSpan(a, i, jlo, jhi);
  }
}

// The rows [2, ni) owned by rank p as an indexed datatype over the grid.
Datatype Type1b(int ni, int nj, const std::vector<int>& bounds, int p) {
  std::vector<int> lens, disps;
  for (int i = 2; i < ni; ++i) {
    auto [jlo, jhi] = Cols1b(i, nj - 3, bounds, p);
    if (jhi > jlo) {
      lens.push_back(jhi - jlo);
      disps.push_back(i * nj + jlo);
    }
  }
  MPI_Datatype t;
  MPI_Type_indexed(static_cast<int>(lens.size()), lens.data(), disps.data(),
                   MPI_DOUBLE, &t);
  return Datatype(t);
}

void Mpi1b(Grid& a, double& comp_s) {
  int ni = a.ni(), nj = a.nj();
  double t0 = MPI_Wtime();
  if (ni <= 2 || nj <= 3) {
    comp_s = 0;
    return;
  }
  std::vector<int> bounds = Split1b(ni, nj);
  for (int i = 2; i < ni; ++i) {
    auto [jlo, jhi] = Cols1b(i, nj - 3, bounds, rank);
    loops::V1bRow(a.Row(i), a.Row(i - 2), jlo, jhi);
  }
  comp_s = MPI_Wtime() - t0;

  if (rank == 0) {
    std::vector<MPI_Request> reqs(nprocs - 1);
    std::vector<Datatype> types;
    types.reserve(nprocs - 1);
    for (int p = 1; p < nprocs; ++p) {
      types.push_back(Type1b(ni, nj, bounds, p));
      MPI_Irecv(a.data(), 1, types.back().get(), p, 0, MPI_COMM_WORLD,
                &reqs[p - 1]);
    }
    MPI_Waitall(nprocs - 1, reqs.data(), MPI_STATUSES_IGNORE);
  } else {
    Datatype t = Type1b(ni, nj, bounds, rank);
    MPI_Send(a.data(), 1, t.get(), 0, 0, MPI_COMM_WORLD);
  }
}

struct MpiVariant {
  std::string_view name;
  void (*init)(Grid& a);
  void (*fn)(Grid& a, double& comp_s);
};

constexpr std::array kVariants{
    MpiVariant{"ref", InitRef, MpiRef},
    MpiVariant{"1b", Init1b, Mpi1b},
};

int Run(const loops::Options& o) {
  auto it =
      std::find_if(kVariants.begin(), kVariants.end(),
                   [&o](const MpiVariant& v) { return v.name == o.variant; });
  if (it == kVariants.end()) {
    if (rank == 0) {
      std::fprintf(stderr, "unknown variant '%s'; available:",
                   std::string(o.variant).c_str());
      for (const MpiVariant& v : kVariants) {
        std::fprintf(stderr, " %s", std::string(v.name).c_str());
      }
      std::fprintf(stderr, "\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const MpiVariant& v = *it;
  if (static_cast<std::int64_t>(o.ni) * o.nj > INT_MAX) {
    if (rank == 0) std::fprintf(stderr, "grid too large for int MPI counts\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Grid a(o.ni, o.nj);
  double min_s = 0, total_s = 0, min_comp = 0;
  for (int r = 0; r < o.reps; ++r) {
    if (rank == 0) {
      loops::Init(a);
    } else {
      v.init(a);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime(), comp;
    v.fn(a, comp);
    double t[2] = {MPI_Wtime() - t0, comp};
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : t, t, 2, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    total_s += t[0];
    if (r == 0 || t[0] < min_s) min_s = t[0];
    if (r == 0 || t[1] < min_comp) min_comp = t[1];
  }

  int status = 0;
  if (rank == 0) {
    if (o.header) std::printf("%s\n", loops::kCsvHeader);
    std::printf("%s,mpi,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                std::string(v.name).c_str(), o.ni, o.nj, nprocs, o.reps, min_s,
                total_s / o.reps, min_comp);
    std::fflush(stdout);
    if (o.out && !loops::Write(o.out, a)) status = 1;
    if (o.check && loops::CheckAgainstSeq(v.name, a) != 0) status = 1;
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Session session(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::optional<loops::Options> o = loops::ParseOptions(argc, argv);
  if (!o) MPI_Abort(MPI_COMM_WORLD, 1);
  return Run(*o);
}
