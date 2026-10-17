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

C++20 (`lab_1`) and C17 (`lab_2`), CMake 3.20 or newer, one top-level project; each lab is an `add_subdirectory`. C++ code manages every resource with RAII (owning types such as `std::unique_ptr`, `std::vector` and the labs' own buffer classes); no manual `malloc`/`free` or `new`/`delete` pairs.

```sh
cmake -S . -B build                    # Release and -march=native by default
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lab_1/test_matmul <variant>    # test a single variant
./build/lab_1/bench -v <variant> -r 5 256 512 1024 > results.csv
OMP_NUM_THREADS=4 ./build/lab_1/bench -v <variant>
```

CI (`.github/workflows/build-test.yml`) runs on a macOS arm64 runner on every PR and on pushes to `master`. It configures with `-DPP_WERROR=ON` (adds `-Werror`), builds, and runs ctest, so new code must compile without warnings and every new test must be registered with ctest. MPI tests run with oversubscription enabled because the runner has few cores.

`-DPP_NATIVE=OFF` disables `-march=native`. OpenMP is required. On macOS the top-level CMakeLists finds Homebrew's keg-only `libomp` via `brew --prefix libomp` (override with `-DOpenMP_ROOT=...`).

## Target machine

Apple M3 Pro, **arm64** (5 performance cores with 128 KiB L1d and a shared 16 MiB L2, 6 efficiency cores with 64 KiB L1d and a shared 4 MiB L2, 18 GiB RAM). The manuals' SIMD examples use x86 SSE/AVX (`<x86intrin.h>`), which does not compile here. Use NEON instead (`<arm_neon.h>`, 128-bit `float32x4_t`, for example `vfmaq_f32` and `vaddvq_f32`). Because the performance and efficiency cores differ, speedup does not grow linearly with thread count. Always report the thread count with results.

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
- `include/matmul.hpp` (namespace `matmul`) defines the RAII types `Buffer` (aligned float array) and `Matrix`, and the variant contract: `void fn(std::size_t n, const float *a, const float *b, float *c)` on row-major n x n floats. A variant must not modify A or B, must overwrite every element of C, and must handle **any n** (pad internally if the algorithm needs 2^k). Scratch buffers such as transposes and Strassen temporaries are allocated inside the variant so their cost is included in the timing.
- `src/variants.cpp` holds the registry. To add a variant, create `src/<name>.cpp`, declare it in `matmul.hpp`, add an entry to `kVariants`, and list the file in `lab_1/CMakeLists.txt`. The bench and tests then pick it up automatically. Entry 0 (`naive`) is the reference result for the tests.
- `tests/test_matmul.cpp` checks every variant. All-ones inputs must give exactly `n`. Random inputs in [0, 1) must match the reference within a relative tolerance of 1e-3. The test sizes include values that are not powers of two.
- `bench/bench.cpp` prints CSV to stdout: `variant,n,threads,reps,min_s,mean_s,gflops`. Timing uses `omp_get_wtime()`. CSV files are gitignored except `lab_*/results/*.csv`.
- Variants, in registry order: `naive`, `transposed`, `blocked` (64 x 64 tiles), `strassen` (cutoff 64 on top of the blocked kernel), `blocked_omp`, `simd` (packed NEON 8 x 12 micro-kernel), `simd_omp`, `best` (`simd_omp`, serial below n = 256, one Strassen level on top for n >= 4096). `naive.c`, `transposed.c` and `blocked.c` are built with auto-vectorization disabled so that the SIMD step stays separate, and with `-falign-loops=64` (otherwise one copy of the blocked inner loop ran twice as slow). The internal kernels with row strides (`gemm_blocked`, `gemm_neon`, `gemm_neon_omp`, `strassen_mul`) are declared in `src/gemm.h`.
- Results: `results/sizes.csv` (every variant; serial ones run with `OMP_NUM_THREADS=1`, parallel ones with 11 threads) and `results/scaling.csv` (parallel variants with 1 to 11 threads) hold the raw bench output of several runs. `plots.ipynb` takes the minimum time per configuration and writes the PNGs into `results/`; re-run it with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab_1/plots.ipynb`. `README.md` is the report.

## Lab 2: loop parallelization (`lab_2/`)

Assignment: for the manual's reference loop (`a[i][j] = sin(2*a[i][j])`) and the individual variants **1b**, **2b** and **3b** (Latin "b" everywhere; the manual writes the Cyrillic letter), compute the distance and direction vectors and the dependence type, write a sequential version and a parallel version, check that both write identical result files (`"%f "` per element, one row per line), and measure time, speedup and efficiency against the number of workers. The reference loop must be parallelized with both OpenMP and MPI; the variants use the technology the manual names: 1b `a[i][j] = sin(5*a[i-2][j+3])` with MPI, 2b `a[i][j] = sin(0.2*a[i+1][j-6])` with OpenMP, 3b (`a = sin(0.01*a)`, then `b[i][j] = a[i+1][j-3]*2`) with OpenMP. Only the loop nest is timed (not the initialization or the file output). Full grade needs ISIZE = JSIZE = 5000, which needs heap allocation. `lab_2/README.md` holds the analysis, results and answers to the control questions.

Structure:
- `include/loops.h` has the loop bodies as `static inline` row functions shared by all implementations, so results match bit for bit (the tests compare bits, not a tolerance). `src/seq.c` has the sequential versions and the `seq_variants[]` registry.
- Three programs with the same options (`-v variant -n ISIZE -m JSIZE -r reps -t threads -o file -c -H`): `lab2_seq` (ref, 1b, 2b, 3b), `lab2_omp` (ref, 2b, 2b-cols, 3b), `lab2_mpi` (ref, 1b). `-c` compares the result with the sequential version in-process, `-o` writes the result file. Each prints one CSV row: `variant,impl,isize,jsize,workers,reps,min_s,mean_s,comp_s` (`comp_s` is MPI time without collecting the result on rank 0).
- ctest runs `-c` for several worker counts and sizes and `tests/compare_files.cmake` (sequential vs parallel result files, as the manual prescribes).
- `scripts/bench.sh [variant...]` writes `lab_2/results/<variant>.csv` (sizes 1000 and 5000, workers 1 2 4 6 8 11). Set `LOCK=<cmd>` to wrap each run in a lock on a shared machine. `plots.ipynb` reads the CSVs and saves `results/<variant>.png`; it is committed with outputs.

```sh
ctest --test-dir build -R lab_2 --output-on-failure
./build/lab_2/lab2_omp -v 2b -t 6 -c
mpirun -np 4 ./build/lab_2/lab2_mpi -v 1b -n 1000 -m 1000 -o par.txt
./build/lab_2/lab2_seq -v 1b -n 1000 -m 1000 -o seq.txt && cmp seq.txt par.txt
lab_2/scripts/bench.sh 1b
.venv/bin/pip install jupyter matplotlib pandas   # venv at the repo root, shared
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab_2/plots.ipynb
```

MPI: Open MPI 5 from Homebrew, found with `find_package(MPI)`; tests launch through `${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} N`. Locally `mpirun` needs no network or host options. It allows as many processes as cores (11); more need `--oversubscribe`. Every rank allocates the whole grid but touches only its own part, because 11 fully initialized 200 MB copies push the machine into swap and ruin the timings. `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` reports 5 performance and 6 efficiency cores, so static equal splits wait for the slow cores above 5 workers; the OpenMP versions use dynamic scheduling.
