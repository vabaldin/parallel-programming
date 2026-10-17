# Lab 1: matrix multiplication

C = A * B for square row-major `float` matrices, in eight versions that follow
the steps of the assignment, timed for n = 64, 128, ..., 4096.

## Variants

| variant | step | idea |
|---|---|---|
| `naive` | 1 | `i, j, k` triple loop; the inner loop walks a column of B with stride n |
| `transposed` | 2 | B is transposed first (inside the timed call), so both operands are read row by row |
| `blocked` | 3 | 64 x 64 tiles of A, B and C, loop order `ii, kk, jj, i, k, j` |
| `strassen` | 4 | Strassen recursion down to n <= 64, then the `blocked` kernel |
| `blocked_omp` | 5 | `blocked` with OpenMP: each thread takes whole 64-row block rows of C (`schedule(dynamic, 1)`) |
| `simd` | 6 | packed GEMM with an 8 x 12 NEON register tile, one thread |
| `simd_omp` | 5 + 6 | `simd` with OpenMP over 64 x 192 tiles of C (`schedule(dynamic, 1)`) |
| `best` | all | `simd_omp`; one thread below n = 256; one Strassen level on top for n >= 4096 |

Code (C++20): `src/naive.cc`, `src/transposed.cc`, `src/blocked.cc`
(`blocked` and `blocked_omp`), `src/strassen.cc`, `src/simd.cc` (`simd` and
`simd_omp`), `src/best.cc`. The kernels behind the variants take row strides,
so Strassen can call them on quadrants without copying (`src/gemm.h`). All
memory is owned by RAII objects: `matmul::Buffer` (an aligned array freed by
`std::unique_ptr`) and `matmul::Matrix`. Scratch memory such as transposed B,
packed panels and Strassen temporaries is a local `Buffer`, so it is released
on every exit path.

### Blocking (step 3)

The tile size has to let three tiles stay in L1d together. The efficiency cores
have the smallest L1d, 64 KiB, and 3 * 64 * 64 * 4 B = 48 KiB fits there; on the
performance cores (128 KiB) there is room to spare. Tiles of 128 (192 KiB)
would not fit in either. A sweep with one thread (seconds, best of 3 runs of 3
repetitions):

| tile | 512 | 1024 | 2048 |
|---:|---:|---:|---:|
| 16 | 0.0511 | 0.454 | 4.31 |
| 32 | 0.0409 | 0.335 | 3.18 |
| **64** | **0.0389** | **0.314** | 3.08 |
| 128 | 0.0426 | 0.343 | 2.92 |
| 256 | 0.0406 | 0.327 | 2.68 |

64 is the best tile up to 1024. At 2048 a 256 tile (768 KiB, held in L2) is 15%
faster. The scalar kernel does about one FMA per cycle, so it barely depends on
memory; the larger tile wins there only because its inner loop is longer. The
spread over the useful tiles is small for the same reason: blocking cuts memory
traffic, but scalar code is bound by arithmetic. The gain of `blocked` over
`transposed` (2.5x to 3x) comes mostly from the loop order: in the `k, j` order the
inner loop updates independent elements of C, while the dot product of
`transposed` is one chain of dependent FMAs (4 cycles each).

### Strassen (step 4)

Each level replaces 8 half-size products by 7, plus 18 additions of half-size
matrices. The products are accumulated straight into the quadrants of C, so a
level needs only three h x h temporaries; the workspace for the whole recursion
is allocated once. A size that does not halve evenly down to the cutoff is
padded with zeros: n = 513 is padded to 520 = 65 * 2^3.
The cutoff was measured with one thread (seconds, best of 3 runs of 3
repetitions):

| cutoff | 512 | 1024 | 2048 |
|---:|---:|---:|---:|
| 32 | 0.0254 | 0.179 | 1.26 |
| **64** | 0.0265 | 0.185 | 1.31 |
| 128 | 0.0292 | 0.209 | 1.47 |
| 256 | 0.0330 | 0.235 | 1.67 |

32 is about 4% faster than 64, but its maximum relative difference from the
classical product at n = 4096 is 1.1e-4, against 3.9e-5 for 64, so 64 was
chosen. The cutoff is that low because the additions are vectorized by the
compiler while the base kernel is scalar.

### SIMD (step 6)

The manual's SIMD examples use SSE and AVX intrinsics from `<x86intrin.h>`.
They do not apply to this machine: the M3 Pro is arm64, so the code uses NEON
intrinsics from `<arm_neon.h>` (128-bit `float32x4_t`, four floats per register).
A dot product over the transposed matrix, as in the manual, would still need a
horizontal sum for each element of C. Instead the kernel keeps an 8 x 12 block of
C in 24 of the 32 NEON registers. Each step of k loads 8 values of A and 12 of B
and issues 24 `vfmaq_laneq_f32`, each multiplying 4 elements of a B row by one
element of A. This is the GotoBLAS scheme:

- A is packed into panels of 8 rows and B into panels of 12 columns, so that the
  kernel reads both sequentially. Rows and columns past n are zero, and edge
  tiles go through a small buffer, so any n works.
- KC = 512: a 512 x 12 panel of B (24 KiB) stays in L1 while the kernel sweeps
  the MC x KC block of A (64 x 512 = 128 KiB, from L2).
- KC, MC and NC (512, 64, 192) were tuned by a sweep over KC in {256, 512},
  MC in {64, 128, 256} and NC in {192, 384, 768}.

One performance core peaks at about 4 FMA units * 4 lanes * 2 flops * 4.05 GHz
= 130 GFLOPS. `simd` reaches 112 GFLOPS, 86% of that.

### OpenMP (step 5) and false sharing

`blocked_omp` hands out 64-row block rows of C, and `simd_omp` hands out
64 x 192 tiles of C. Each piece of C is written by one thread only, and there are
no shared counters or reductions. Two threads can only share a 128-byte cache
line of C at the border of two pieces: one line per row of a tile border at
most, against 64 x 192 floats written inside the tile. For a power-of-two n the
borders fall on multiples of 64 rows and of 192 columns (768 B = 6 lines), so
with a line-aligned C nothing is shared at all. In `simd_omp` the packing of A
and B is split between the threads as well, and each thread reads the shared
packed buffers only.

Static scheduling would give every thread the same number of tiles, and then
the efficiency cores would finish last. `schedule(dynamic, 1)` lets the
performance cores take more tiles. At n = 4096 there are 64 * 22 = 1408 tiles,
enough to balance 11 threads.

### best

`best` is `simd_omp` (blocking + NEON + OpenMP) with two size switches, both
measured. A parallel call has about 100 us of overhead, more than the whole
serial product for n <= 128, so `best` runs on one thread below 256. At
n = 4096 one Strassen level on top of `simd_omp` saves 1/8 of the
multiplications and is 6% faster; at 2048 and below the extra additions, which
only use memory bandwidth, cost more than that.

## Method

- Machine: Apple MacBook Pro with M3 Pro (arm64), 5 performance cores
  (128 KiB L1d each, shared 16 MiB L2) and 6 efficiency cores (64 KiB L1d each,
  shared 4 MiB L2), 128-byte cache lines, 18 GiB RAM, macOS 26.4.1. The data
  below were measured on AC power with nothing else running.
- Compiler: Homebrew clang 20.1.8 with Homebrew libomp, CMake Release build:
  `-O3 -DNDEBUG -std=c++20 -march=native -Wall -Wextra -fopenmp=libomp`.
  `naive.cc`, `transposed.cc` and `blocked.cc` are also built with
  `-fno-vectorize -fno-slp-vectorize`, so that steps 1 to 4 stay scalar and the
  SIMD step shows its own effect. Without that flag the compiler would
  vectorize the `k, j` loop of `blocked` by itself.
- These three files also get `-falign-loops=64`. Without it, `blocked_omp` on one
  thread was twice as slow as `blocked`, although both inline the same inner
  loop: in the OpenMP copy the 6-instruction loop body crossed a 64-byte boundary.
- The tables and plots were measured with the first version of the code,
  written in C17. The C++20 rewrite keeps the same kernels and loop order;
  spot checks (naive, transposed, blocked, strassen, blocked_omp and simd
  on one thread at 1024, simd at 2048, and the parallel variants at 1024,
  2048 and 4096 with 11 threads) agree with the tables within 4%.
- Inputs are random values in [0, 1). Every call allocates its own scratch
  memory (transposed B, packed panels, Strassen temporaries), so that cost is
  included in the time.
- Timing: `bench` takes `omp_get_wtime()` around each call. Each configuration
  was run 3 times, with 10 repetitions per run up to n = 512 (and at 1024 for
  the parallel variants), 5 at 1024 for the serial ones, 5 at 2048 and 3 at
  4096 for the parallel ones, and 3 at 2048 and 4096 for `simd`. The slow
  scalar cases ran fewer times: `transposed`, `blocked` and `strassen` 2 runs
  of 3 repetitions at 2048, `naive` 1 repetition at 2048, and all four
  1 repetition at 4096. In the scaling runs `blocked_omp` had 2 runs of 3
  repetitions at 2048 and 1 repetition at 4096. The tables give the minimum
  over all repetitions.
  GFLOPS = 2 n^3 / time.
- Thread counts: the serial variants ran with `OMP_NUM_THREADS=1`, the parallel
  ones with all 11 cores. The scaling runs used `OMP_NUM_THREADS` = 1, 2, 4, 6, 8
  and 11 at n = 2048 and 4096. S = T1 / Tp and E = S / p, where T1 is the same
  variant on one thread.
- Correctness: `matmul_test` compares every variant with `naive` on random inputs
  (relative tolerance 1e-3) and checks all-ones inputs exactly, for n = 1, 2, 7,
  16, 64, 100, 128, 256, 300 and 513. The paths that only run for large n were
  also checked against `simd_omp` at n = 4096: `blocked` matches bit for bit,
  `strassen` within 3.9e-5 and `best` within 9.3e-6, and all-ones inputs give
  exactly n.

Raw data: `results/sizes.csv` and `results/scaling.csv` (one line per bench
run, with the thread count in the `threads` column). Plots: `plots.ipynb`.

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
OMP_NUM_THREADS=1 ./build/lab_1/bench -v simd -r 5 1024 2048
OMP_NUM_THREADS=11 ./build/lab_1/bench -v best -r 3 4096
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab_1/plots.ipynb
```

## Results

Time (p is the number of threads):

| variant | p | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 1 | 108 us | 1.18 ms | 11.31 ms | 105.61 ms | 955.51 ms | 8.63 s | 206 s |
| transposed | 1 | 105 us | 1.11 ms | 10.88 ms | 104.64 ms | 911.48 ms | 7.71 s | 63.40 s |
| blocked | 1 | 70 us | 567 us | 4.72 ms | 38.72 ms | 312.26 ms | 3.09 s | 23.03 s |
| strassen | 1 | 70 us | 503 us | 3.77 ms | 26.81 ms | 188.51 ms | 1.33 s | 9.42 s |
| blocked_omp | 11 | 87 us | 318 us | 1.28 ms | 8.22 ms | 50.50 ms | 440.16 ms | 3.32 s |
| simd | 1 | 7 us | 46 us | 315 us | 2.42 ms | 19.12 ms | 153.51 ms | 1.24 s |
| simd_omp | 11 | 117 us | 120 us | 172 us | 530 us | 3.10 ms | 23.34 ms | 187.31 ms |
| best | 11 | 8 us | 49 us | 176 us | 526 us | 3.12 ms | 23.33 ms | 175.98 ms |

GFLOPS:

| variant | p | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 1 | 4.9 | 3.6 | 3.0 | 2.5 | 2.2 | 2.0 | 0.7 |
| transposed | 1 | 5.0 | 3.8 | 3.1 | 2.6 | 2.4 | 2.2 | 2.2 |
| blocked | 1 | 7.5 | 7.4 | 7.1 | 6.9 | 6.9 | 5.6 | 6.0 |
| strassen | 1 | 7.5 | 8.3 | 8.9 | 10.0 | 11.4 | 12.9 | 14.6 |
| blocked_omp | 11 | 6.0 | 13.2 | 26.3 | 32.7 | 42.5 | 39.0 | 41.3 |
| simd | 1 | 73.3 | 91.6 | 106.5 | 111.1 | 112.3 | 111.9 | 110.7 |
| simd_omp | 11 | 4.5 | 34.9 | 195.2 | 506.5 | 692.5 | 736.2 | 733.7 |
| best | 11 | 64.7 | 85.8 | 190.4 | 510.4 | 689.2 | 736.3 | 781.0 |

For Strassen, GFLOPS still counts 2 n^3 operations, so it is an effective rate.

![time vs size](results/time_vs_size.png)

![GFLOPS vs size](results/gflops_vs_size.png)

Speedup and efficiency, written as S / E:

| variant | n | T(1) | T(11) | p=2 | p=4 | p=6 | p=8 | p=11 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| blocked_omp | 2048 | 3.09 s | 450.16 ms | 1.81 / 0.91 | 3.61 / 0.90 | 4.76 / 0.79 | 5.97 / 0.75 | 6.87 / 0.62 |
| blocked_omp | 4096 | 23.04 s | 3.38 s | 1.93 / 0.96 | 3.79 / 0.95 | 5.03 / 0.84 | 5.88 / 0.74 | 6.81 / 0.62 |
| simd_omp | 2048 | 152.66 ms | 23.24 ms | 1.94 / 0.97 | 3.77 / 0.94 | 5.00 / 0.83 | 5.67 / 0.71 | 6.57 / 0.60 |
| simd_omp | 4096 | 1.24 s | 186.10 ms | 1.95 / 0.98 | 3.81 / 0.95 | 5.05 / 0.84 | 5.81 / 0.73 | 6.67 / 0.61 |
| best | 2048 | 152.23 ms | 23.32 ms | 1.94 / 0.97 | 3.76 / 0.94 | 4.97 / 0.83 | 5.66 / 0.71 | 6.53 / 0.59 |
| best | 4096 | 1.09 s | 175.17 ms | 1.93 / 0.97 | 3.68 / 0.92 | 4.83 / 0.80 | 5.44 / 0.68 | 6.23 / 0.57 |

![speedup](results/speedup.png)

![efficiency](results/efficiency.png)

## Analysis

- **naive vs transposed.** Up to 2048 the two are equally slow (about 2 GFLOPS):
  both are limited by the chain of dependent FMAs in the scalar sum, and the
  hardware prefetcher hides the strided reads of B as long as the touched lines
  of B fit in the caches. At 4096 a column of B spans 4096 lines 16 KiB apart,
  so `naive` drops to 0.7 GFLOPS while `transposed` stays at 2.2 (3.2x faster).
- **blocked** is 2.5x to 3x faster than `transposed` (see above) and stays at
  5.6 to 7.5 GFLOPS for all sizes, so the tiles keep the data in cache.
- **strassen** has the expected O(n^2.81) slope: every doubling of n multiplies
  the time by about 7 instead of 8, and at 4096 it is 2.4x faster than its own
  base kernel.
- **simd** is 16x to 20x faster than `blocked`, more than the 4 lanes of a
  register: the register tile turns the kernel from one FMA per cycle into four
  independent 4-lane FMAs per cycle, and packing makes every load sequential.
  It is flat at 110 GFLOPS from n = 512 on, so the blocking for L1 and L2 works
  for every size.
- **Parallel overhead.** `simd_omp` takes about 0.1 ms even at n = 64 and
  128, where the serial `simd` needs 7 to 46 us; the threads pay off from
  n = 256 on. This is why `best` stays serial below 256.
- **best** reaches 736 GFLOPS at 2048 (23.3 ms) and 781 effective GFLOPS at
  4096 (176 ms), 1170x faster than `naive` at 4096.

### Where the speedup saturates and why

All three parallel variants scale almost linearly up to 4 threads
(E = 0.90 to 0.98) and saturate past 5 threads, at S = 6.2 to 6.9 for 11
threads (E = 0.57 to 0.62):

- The machine has only 5 performance cores. The scheduler puts the first
  threads on them, and every thread past the fifth runs on an efficiency
  core. At p = 6,
  S is still about 5, and the last 5 threads together add only 1.4 to 2.1
  more. An efficiency core is worth about a third of a performance core for
  this code: it has a lower clock, a narrower core and half the L1d.
- Measured against the available compute rather than the core count,
  E = 6.7 / (5 + 6 / 3) = 0.95 for `simd_omp`. The algorithm itself scales well:
  the work splits into independent tiles without communication, and dynamic
  scheduling feeds the fast cores more of them. The ceiling is the asymmetric
  hardware.
- `best` scales worse than `simd_omp` at 4096 (6.2 against 6.7) because its
  Strassen additions only stream memory, and memory bandwidth stops growing
  long before all 11 cores are busy. Its absolute time is still the lowest.
- `blocked_omp` has the highest S, because its scalar per-core speed is far
  from any shared limit, but it is about 19 times slower than `best`.

## Control questions

1. **Complexity of the basic algorithm.** n^3 multiply-add pairs, O(n^3) time
   (2 n^3 floating-point operations), with O(n^2) data.
2. **Fast matrix multiplication.** Strassen (1969) multiplies 2 x 2 block
   matrices with 7 instead of 8 block products, giving O(n^log2(7)) =
   O(n^2.807). Winograd's variant needs 15 instead of 18 additions.
   Coppersmith-Winograd and its later refinements (Le Gall, Alman and Vassilevska
   Williams and others) bring the exponent down to about 2.37, but their constants
   are so large that they are never used in practice. Strassen pays off only for
   large n, on top of a fast classical kernel, and it is somewhat less accurate
   numerically.
3. **Cache memory.** A small fast memory between the cores and DRAM, organized
   in lines (128 bytes here) and levels (L1, L2, system cache). A hit costs a few
   cycles, a miss to DRAM about a hundred nanoseconds. Matrix multiplication
   does O(n^3) work on O(n^2) data, so each element can be reused about n times,
   but only if it is still in the cache when it is used again. The naive loop
   reads B by columns and uses one float of every line it loads; at large n the
   lines are evicted before the next column, which lies in the same lines, needs
   them. Transposition, loop reordering and
   blocking turn these accesses into sequential ones and keep the reused data in
   the cache.
4. **Blocked algorithm.** Split the matrices into b x b blocks and compute
   C_IJ += A_IK * B_KJ block by block, with b chosen so that the blocks being
   used fit in the cache level being targeted. Every element loaded from memory
   is then used b times instead of once. Real implementations block several
   times: for registers (the 8 x 12 micro-tile), for L1 (a panel of B) and for
   L2 (a block of A), and they pack the blocks into contiguous buffers.
5. **Strassen algorithm.** Split A, B and C into four quadrants and compute
   M1 = (A11 + A22)(B11 + B22), M2 = (A21 + A22) B11, M3 = A11 (B12 - B22),
   M4 = A22 (B21 - B11), M5 = (A11 + A12) B22, M6 = (A21 - A11)(B11 + B12),
   M7 = (A12 - A22)(B21 + B22). Then C11 = M1 + M4 - M5 + M7, C12 = M3 + M5,
   C21 = M2 + M4 and C22 = M1 - M2 + M3 + M6. The seven products are computed
   recursively; below a cutoff size a classical kernel is faster. Sizes that do
   not halve evenly are padded with zeros.
6. **Parallelization.** Yes. Every element (or block) of C can be computed
   independently, so the rows or tiles of C can be split between threads with
   no synchronization except a final barrier, and with read-only sharing of A
   and B. Care is needed with false sharing on the borders of the pieces of C,
   with load balance (dynamic scheduling here, because the cores are uneven),
   and with memory bandwidth, which all cores share. The seven Strassen
   products and its matrix additions can also run in parallel.
7. **SIMD.** One instruction processes a vector of floats: 4 in a 128-bit NEON
   or SSE register, 8 in AVX. The useful form is an outer-product kernel: load a
   short row of B into vector registers, broadcast one element of A (on NEON a
   lane of a register, `vfmaq_laneq_f32`), and fused-multiply-add into a block of
   C that stays in registers for the whole k loop. Enough independent
   accumulators are needed to cover the FMA latency, and the operands should be
   packed so the loads are sequential and aligned. The dot product form of the
   manual also works, but needs a horizontal sum for every element of C.
8. **Speedup and efficiency.** Speedup S(p) = T(1) / T(p): how many times p
   threads are faster than one. Efficiency E(p) = S(p) / p: the fraction of the
   ideal linear speedup that is achieved, at most 1 in the usual case. Amdahl's
   law bounds S by 1 / (f + (1 - f) / p) for a serial fraction f. On a machine
   with uneven cores, like this one, E also falls because the extra cores are
   slower, not only because of the overhead.
