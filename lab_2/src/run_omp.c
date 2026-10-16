/* OpenMP versions of ref, 2b and 3b. Usage: see options_parse; -t sets the
 * thread count (default OMP_NUM_THREADS). */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loops.h"

/* The cores are not equal (5 performance, 6 efficiency), so rows are handed
 * out dynamically in small chunks instead of in equal static blocks. */
#define ROW_CHUNK 8
#define BLOCKS_PER_THREAD 8

static void par_ref(int ni, int nj, double *a, double *b) {
    (void)b;
#pragma omp parallel for schedule(dynamic, ROW_CHUNK)
    for (int i = 0; i < ni; ++i)
        ref_row(&AT(a, nj, i, 0), nj);
}

static int block_lo(int rows, int nb, int k) {
    return (int)((long long)rows * k / nb);
}

/* Anti-dependence (i, j) -> (i+1, j-6): row i must read row i+1 before row
 * i+1 is overwritten. The rows are cut into blocks that run in any order,
 * each in sequential order inside. The only row a block reads but does not
 * own is the first row of the next block, so copies of those rows are taken
 * before any block starts (the barrier of the first omp for). */
static void par_2b(int ni, int nj, double *a, double *b) {
    (void)b;
    int rows = ni - 1;
    if (rows <= 0 || nj <= 6)
        return;
    int nb = BLOCKS_PER_THREAD * omp_get_max_threads();
    if (nb > rows)
        nb = rows;
    size_t w = (size_t)(nj - 6);
    double *halo = malloc((size_t)nb * w * sizeof(double));
    if (!halo)
        abort();
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int k = 0; k < nb; ++k)
            memcpy(halo + k * w, &AT(a, nj, block_lo(rows, nb, k + 1), 0),
                   w * sizeof(double));
#pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < nb; ++k) {
            int lo = block_lo(rows, nb, k), hi = block_lo(rows, nb, k + 1);
            for (int i = lo; i < hi - 1; ++i)
                v2b_row(&AT(a, nj, i, 0), &AT(a, nj, i + 1, 0), 6, nj);
            v2b_row(&AT(a, nj, hi - 1, 0), halo + k * w, 6, nj);
        }
    }
    free(halo);
}

/* Alternative for 2b: the inner loop has no dependence, so split j and
 * keep i sequential. The barrier at the end of each omp for orders the reads
 * of row i+1 before its writes. */
static void par_2b_cols(int ni, int nj, double *a, double *b) {
    (void)b;
#pragma omp parallel
    for (int i = 0; i < ni - 1; ++i) {
        double *a_i = &AT(a, nj, i, 0);
        const double *a_ip1 = &AT(a, nj, i + 1, 0);
#pragma omp for schedule(static)
        for (int j = 6; j < nj; ++j)
            v2b_row(a_i, a_ip1, j, j + 1);
    }
}

/* Both nests are free of loop-carried dependences. The second one reads
 * a[i+1] produced by the first, so the barrier between them is required. */
static void par_3b(int ni, int nj, double *a, double *b) {
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, ROW_CHUNK)
        for (int i = 0; i < ni; ++i)
            v3b_row_a(&AT(a, nj, i, 0), nj);
#pragma omp for schedule(static)
        for (int i = 0; i < ni - 1; ++i)
            v3b_row_b(&AT(b, nj, i, 0), &AT(a, nj, i + 1, 0), nj);
    }
}

struct omp_variant {
    struct loop_variant v;
    const char *seq;
};

static const struct omp_variant variants[] = {
    {{"ref", 0, par_ref}, "ref"},
    {{"2b", 0, par_2b}, "2b"},
    {{"2b-cols", 0, par_2b_cols}, "2b"},
    {{"3b", 1, par_3b}, "3b"},
};
#define NVARIANTS (sizeof(variants) / sizeof(variants[0]))

int main(int argc, char **argv) {
    struct options o;
    if (options_parse(argc, argv, &o) != 0)
        return 1;
    const struct omp_variant *ov = NULL;
    for (size_t i = 0; i < NVARIANTS; ++i)
        if (strcmp(variants[i].v.name, o.variant) == 0)
            ov = &variants[i];
    if (!ov) {
        fprintf(stderr, "unknown variant '%s'; available:", o.variant);
        for (size_t i = 0; i < NVARIANTS; ++i)
            fprintf(stderr, " %s", variants[i].v.name);
        fprintf(stderr, "\n");
        return 1;
    }
    const struct loop_variant *v = &ov->v;
    if (o.threads > 0)
        omp_set_num_threads(o.threads);

    double *a = grid_alloc(o.ni, o.nj);
    double *b = v->uses_b ? grid_alloc(o.ni, o.nj) : NULL;
    double min_s = 0, total_s = 0;
    for (int r = 0; r < o.reps; ++r) {
        grid_init(o.ni, o.nj, a, b);
        double t0 = omp_get_wtime();
        v->fn(o.ni, o.nj, a, b);
        double dt = omp_get_wtime() - t0;
        total_s += dt;
        if (r == 0 || dt < min_s)
            min_s = dt;
    }

    if (o.header)
        printf("%s\n", CSV_HEADER);
    printf("%s,omp,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", v->name, o.ni, o.nj,
           omp_get_max_threads(), o.reps, min_s, total_s / o.reps, min_s);

    const double *result = v->uses_b ? b : a;
    int status = 0;
    if (o.out && grid_write(o.out, o.ni, o.nj, result) != 0)
        status = 1;
    if (o.check && check_against_seq(ov->seq, o.ni, o.nj, result) != 0)
        status = 1;
    grid_free(a);
    grid_free(b);
    return status;
}
