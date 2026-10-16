/* Usage: bench [-v variant]... [-r reps] [n ...]
 * Defaults: all variants, 5 reps, sizes 64 to 4096. gflops uses min_s and
 * 2 * n^3 operations. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "matmul.h"

#define MAX_SELECTED 64

static void run(const struct matmul_variant *v, size_t n, int reps) {
    float *a = matrix_alloc(n), *b = matrix_alloc(n), *c = matrix_alloc(n);
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed for n=%zu\n", n);
        exit(1);
    }
    matrix_fill_random(n, a, 1);
    matrix_fill_random(n, b, 2);

    double min_s = 0.0, total_s = 0.0;
    for (int r = 0; r < reps; ++r) {
        double t0 = omp_get_wtime();
        v->fn(n, a, b, c);
        double dt = omp_get_wtime() - t0;
        total_s += dt;
        if (r == 0 || dt < min_s)
            min_s = dt;
    }

    double gflops = 2.0 * (double)n * n * n / min_s * 1e-9;
    printf("%s,%zu,%d,%d,%.9f,%.9f,%.3f\n", v->name, n, omp_get_max_threads(),
           reps, min_s, total_s / reps, gflops);
    fflush(stdout);

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

int main(int argc, char **argv) {
    const struct matmul_variant *selected[MAX_SELECTED];
    size_t nselected = 0;
    int reps = 5;

    int opt;
    while ((opt = getopt(argc, argv, "v:r:")) != -1) {
        switch (opt) {
        case 'v': {
            const struct matmul_variant *v = matmul_find(optarg);
            if (!v) {
                fprintf(stderr, "unknown variant '%s'; available:", optarg);
                for (size_t i = 0; i < matmul_variants_count; ++i)
                    fprintf(stderr, " %s", matmul_variants[i].name);
                fprintf(stderr, "\n");
                return 1;
            }
            if (nselected < MAX_SELECTED)
                selected[nselected++] = v;
            break;
        }
        case 'r':
            reps = atoi(optarg);
            if (reps < 1)
                reps = 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v variant]... [-r reps] [n ...]\n",
                    argv[0]);
            return 1;
        }
    }
    if (nselected == 0)
        for (size_t i = 0; i < matmul_variants_count && i < MAX_SELECTED; ++i)
            selected[nselected++] = &matmul_variants[i];

    static const size_t default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    enum { NDEFAULT = sizeof(default_sizes) / sizeof(default_sizes[0]) };
    size_t nsizes = (size_t)(argc - optind);
    size_t *sizes = malloc((nsizes ? nsizes : NDEFAULT) * sizeof(*sizes));
    if (nsizes == 0) {
        nsizes = NDEFAULT;
        for (size_t i = 0; i < nsizes; ++i)
            sizes[i] = default_sizes[i];
    } else {
        for (size_t i = 0; i < nsizes; ++i)
            sizes[i] = (size_t)strtoul(argv[optind + i], NULL, 10);
    }

    printf("variant,n,threads,reps,min_s,mean_s,gflops\n");
    for (size_t s = 0; s < nsizes; ++s)
        for (size_t i = 0; i < nselected; ++i)
            run(selected[i], sizes[s], reps);

    free(sizes);
    return 0;
}
