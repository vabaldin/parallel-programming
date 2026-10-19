# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Lab assignments for the MIPT course "Introduction to parallelization of algorithms and programs" (term 7). Each `lab_N/` directory is one assignment. The assignment manuals are Russian PDFs kept locally as `lab_N/assignment.pdf` and gitignored. The summaries below cover what each lab requires, so the PDFs rarely need re-reading.

## Git workflow

Never commit to `master`. For every task: create a new branch, commit, `git push -u origin <branch>`, then open a PR with `gh pr create --assignee vabaldin` (every PR is assigned to `vabaldin`). Related PRs form a stack: if the work depends on an unmerged branch, branch from that branch and open the PR with `--base <parent-branch>` instead of `master`, so that each PR shows only its own commits.

## Plots

Plots are built in a Jupyter notebook per lab (`lab_N/plots.ipynb`). The notebook reads `lab_N/results/*.csv`, also saves PNGs there for the README, and is committed with its outputs. Use the repo-root venv: `python3 -m venv .venv && .venv/bin/pip install jupyter matplotlib pandas`. Re-run with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab_N/plots.ipynb`.

## Code style

- [Google C++ style](https://google.github.io/styleguide/cppguide.html), formatting and naming included: `.cc`/`.h` files, `#define` header guards (`LAB_1_INCLUDE_MATMUL_H_`), `CamelCase` types and functions, `snake_case` variables and parameters, `member_` for private class members, `kName` constants, `UPPER_CASE` macros, `<cstdint>` types instead of `long`/`long long`. Simple accessors (`data`, `size`, `get`, `n`, `ni`, `nj`) keep variable-style names, as the guide allows.
- `.clang-format` (`BasedOnStyle: Google`) and `.clang-tidy` (`google-*` and `readability-identifier-naming`, warnings are errors) enforce it. Check locally with `scripts/check_style.sh build` after configuring (it also rejects non-ASCII text outside the reports); fix formatting with `git ls-files '*.cc' '*.h' | xargs clang-format -i`. Use the clang-format and clang-tidy from Homebrew `llvm@20`, the same version as CI.
- Comments are rare and written in English. Comment only what the code cannot say itself.
- ASCII only in all files: no Unicode punctuation, arrows, math symbols or Cyrillic. The exception are the lab reports, `lab_N/README.md` and `lab_N/plots.ipynb` (markdown, plot titles and labels): they are written in Russian, briefly and to the point, because they are submitted as the report.

## Build and test

C++20, CMake 3.20 or newer, one top-level project; each lab is an `add_subdirectory`. C++ code manages every resource with RAII (owning types such as `std::unique_ptr`, `std::vector` and the labs' own buffer classes); no manual `malloc`/`free` or `new`/`delete` pairs.

```sh
cmake -S . -B build                    # Release and -march=native by default
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lab_1/matmul_test <variant>    # test a single variant
./build/lab_1/bench -v <variant> -r 5 256 512 1024 > results.csv
OMP_NUM_THREADS=4 ./build/lab_1/bench -v <variant>
```

CI (`.github/workflows/build-test.yml`) runs on a macOS arm64 runner on every PR and on pushes to `master`. It uses Homebrew `llvm@20`, configures with `-DPP_WERROR=ON` (adds `-Werror`), builds, runs ctest and `scripts/check_style.sh`, so new code must compile without warnings, pass the style check, and every new test must be registered with ctest. MPI tests run with oversubscription enabled because the runner has few cores.

`-DPP_NATIVE=OFF` disables `-march=native`; `-DPP_WERROR=ON` adds `-Werror`. OpenMP is required. On macOS the top-level CMakeLists finds Homebrew's keg-only `libomp` via `brew --prefix libomp` (override with `-DOpenMP_ROOT=...`).

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
- `include/matmul.h` (namespace `matmul`) defines the RAII types `Buffer` (aligned float array) and `Matrix`, and the variant contract: `void fn(std::size_t n, const float *a, const float *b, float *c)` on row-major n x n floats. A variant must not modify A or B, must overwrite every element of C, and must handle **any n** (pad internally if the algorithm needs 2^k). Scratch buffers such as transposes and Strassen temporaries are allocated inside the variant so their cost is included in the timing.
- `src/variants.cc` holds the registry. To add a variant, create `src/<name>.cc`, declare it in `matmul.h`, add an entry to `kVariants`, and list the file in `lab_1/CMakeLists.txt`. The bench and tests then pick it up automatically. Entry 0 (`naive`) is the reference result for the tests.
- `tests/matmul_test.cc` checks every variant. All-ones inputs must give exactly `n`. Random inputs in [0, 1) must match the reference within a relative tolerance of 1e-3. The test sizes include values that are not powers of two.
- `bench/bench.cc` prints CSV to stdout: `variant,n,threads,reps,min_s,mean_s,gflops`. Timing uses `omp_get_wtime()`. CSV files are gitignored except `lab_*/results/*.csv`.

## Lab 2: loop parallelization (`lab_2/`)

Not started. The manual covers MPI (distributed memory: `mpicc`, `mpirun -np`, point-to-point send and receive, `MPI_Wtime`), OpenMP for parallelizing loops with data dependencies, Amdahl's law, and speedup and efficiency. Open MPI is installed via Homebrew (`mpicc` is on PATH).
