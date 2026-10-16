# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Lab assignments for the MIPT course «Введение в распараллеливание алгоритмов и программ» (term 7). Each `lab_N/` directory is one assignment; the assignment manuals are Russian PDFs kept locally in each lab directory (not committed). Summaries of what each lab requires are below so the PDFs rarely need re-reading.

## Git workflow

Never commit to `master`. For every task: create a new branch off `master`, commit, `git push -u origin <branch>`, then open a PR with `gh pr create --base master`.

## Build & test

C17, CMake ≥ 3.20, single top-level project; each lab is an `add_subdirectory`.

```sh
cmake -S . -B build                    # Release + -march=native by default
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lab_1/test_matmul <variant>    # test a single variant
./build/lab_1/bench -v <variant> -r 5 256 512 1024 > results.csv
OMP_NUM_THREADS=4 ./build/lab_1/bench -v <variant>
```

`-DPP_NATIVE=OFF` disables `-march=native`. OpenMP is required; on macOS the top-level CMakeLists auto-detects Homebrew's keg-only `libomp` via `brew --prefix libomp` (override with `-DOpenMP_ROOT=...`).

## Target machine

Apple M3 Pro, **arm64** (6P+5E cores, 64 KiB L1d, 4 MiB L2 per cluster). The manuals' SIMD examples use x86 SSE/AVX (`<x86intrin.h>`), which does not compile here: use NEON (`<arm_neon.h>`, 128-bit `float32x4_t`, e.g. `vfmaq_f32`, `vaddvq_f32`). Mixed P/E cores make thread-count scaling non-linear; report thread counts with results.

## Lab 1 — matrix multiplication (`lab_1/`)

Assignment: implement and time square matrix multiplication, each step verified to give the same result as the naive version, timed on `float` square matrices of power-of-two sizes 64…4096 (the bench default):
1. naive triple loop; 2. transpose B for sequential access; 3. blocked (cache tiling); 4. Strassen (or another fast algorithm); 5. OpenMP parallelization of one of the versions; 6. SIMD vectorization.
Deliverable: a combined fastest version (target size 2048 or 4096), time-vs-size plots for every step, and speedup/efficiency plots (S = T1/Tp, E = S/p) for the parallel versions.

Structure:
- `include/matmul.h` — the variant contract: `void fn(size_t n, const float *a, const float *b, float *c)`, row-major n×n floats. Variants must not modify A/B, must fully overwrite C, and must handle **any n** (pad internally if an algorithm needs 2^k). Scratch buffers (transposes, Strassen temporaries) are allocated inside the variant so their cost is timed.
- `src/variants.c` — the registry. Adding a variant = new `src/<name>.c`, declaration in `matmul.h`, entry in `matmul_variants[]`, file in `lab_1/CMakeLists.txt`. The bench and tests pick it up automatically. Entry 0 (`naive`) is the reference result for tests.
- `tests/test_matmul.c` — for every variant: all-ones inputs must give exactly `n`, random [0,1) inputs must match the reference within 1e-3 relative tolerance, over sizes including non-powers-of-two.
- `bench/bench.c` — CSV to stdout: `variant,n,threads,reps,min_s,mean_s,gflops`; timing uses `omp_get_wtime()`. CSV files are gitignored.

## Lab 2 — loop parallelization (`lab_2/`)

Not started. Manual covers MPI (distributed memory: `mpicc`, `mpirun -np`, point-to-point send/recv, `MPI_Wtime`) and OpenMP for parallelizing loops with data dependencies, plus Amdahl's law and speedup/efficiency. Open MPI is installed via Homebrew (`mpicc` on PATH).
