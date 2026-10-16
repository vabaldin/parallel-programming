# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Lab assignments for the MIPT course "Introduction to parallelization of algorithms and programs" (term 7). Each `lab_N/` directory is one assignment. The assignment manuals are Russian PDFs kept locally in each lab directory and are gitignored. The summaries below cover what each lab requires, so the PDFs rarely need re-reading.

## Git workflow

Never commit to `master`. For every task: create a new branch, commit, `git push -u origin <branch>`, then open a PR with `gh pr create`. Related PRs form a stack: if the work depends on an unmerged branch, branch from that branch and open the PR with `--base <parent-branch>` instead of `master`, so that each PR shows only its own commits.

## Plots

Plots are built in a Jupyter notebook per lab (`lab_N/plots.ipynb`). The notebook reads `lab_N/results/*.csv`, also saves PNGs there for the README, and is committed with its outputs. Use the repo-root venv: `python3 -m venv .venv && .venv/bin/pip install jupyter matplotlib pandas`. Re-run with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab_N/plots.ipynb`.

## Code style

- Comments are rare and written in English. Comment only what the code cannot say itself.
- ASCII only in all files: no Unicode punctuation, arrows, math symbols or Cyrillic.

## Build and test

C17, CMake 3.20 or newer, one top-level project; each lab is an `add_subdirectory`.

```sh
cmake -S . -B build                    # Release and -march=native by default
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lab_1/test_matmul <variant>    # test a single variant
./build/lab_1/bench -v <variant> -r 5 256 512 1024 > results.csv
OMP_NUM_THREADS=4 ./build/lab_1/bench -v <variant>
```

`-DPP_NATIVE=OFF` disables `-march=native`. OpenMP is required. On macOS the top-level CMakeLists finds Homebrew's keg-only `libomp` via `brew --prefix libomp` (override with `-DOpenMP_ROOT=...`).

## Target machine

Apple M3 Pro, **arm64** (6 performance and 5 efficiency cores, 64 KiB L1d, 4 MiB L2 per cluster). The manuals' SIMD examples use x86 SSE/AVX (`<x86intrin.h>`), which does not compile here. Use NEON instead (`<arm_neon.h>`, 128-bit `float32x4_t`, for example `vfmaq_f32` and `vaddvq_f32`). Because the performance and efficiency cores differ, speedup does not grow linearly with thread count. Always report the thread count with results.

## Lab 1: matrix multiplication (`lab_1/`)

Assignment: implement square matrix multiplication in several versions, check that each gives the same result as the naive version, and time it on `float` matrices of power-of-two sizes from 64 to 4096 (the bench default):
1. naive triple loop
2. transpose B so the inner loop reads memory sequentially
3. blocked (cache tiling)
4. Strassen (or another fast algorithm)
5. OpenMP parallelization of one of the versions
6. SIMD vectorization

Deliverable: a combined fastest version (target size 2048 or 4096), time vs size plots for every step, and speedup and efficiency plots (S = T1/Tp, E = S/p) for the parallel versions.

Structure:
- `include/matmul.h` defines the variant contract: `void fn(size_t n, const float *a, const float *b, float *c)` on row-major n x n floats. A variant must not modify A or B, must overwrite every element of C, and must handle **any n** (pad internally if the algorithm needs 2^k). Scratch buffers such as transposes and Strassen temporaries are allocated inside the variant so their cost is included in the timing.
- `src/variants.c` holds the registry. To add a variant, create `src/<name>.c`, declare it in `matmul.h`, add an entry to `matmul_variants[]`, and list the file in `lab_1/CMakeLists.txt`. The bench and tests then pick it up automatically. Entry 0 (`naive`) is the reference result for the tests.
- `tests/test_matmul.c` checks every variant. All-ones inputs must give exactly `n`. Random inputs in [0, 1) must match the reference within a relative tolerance of 1e-3. The test sizes include values that are not powers of two.
- `bench/bench.c` prints CSV to stdout: `variant,n,threads,reps,min_s,mean_s,gflops`. Timing uses `omp_get_wtime()`. CSV files are gitignored.

## Lab 2: loop parallelization (`lab_2/`)

Not started. The manual covers MPI (distributed memory: `mpicc`, `mpirun -np`, point-to-point send and receive, `MPI_Wtime`), OpenMP for parallelizing loops with data dependencies, Amdahl's law, and speedup and efficiency. Open MPI is installed via Homebrew (`mpicc` is on PATH).
