#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "loops.h"

#define GRID_ALIGN 64

double *grid_alloc(int ni, int nj) {
    size_t bytes = (size_t)ni * (size_t)nj * sizeof(double);
    bytes = (bytes + GRID_ALIGN - 1) / GRID_ALIGN * GRID_ALIGN;
    double *x = aligned_alloc(GRID_ALIGN, bytes ? bytes : GRID_ALIGN);
    if (!x) {
        fprintf(stderr, "allocation of %d x %d grid failed\n", ni, nj);
        exit(1);
    }
    return x;
}

void grid_free(double *x) { free(x); }

void grid_init_span(int nj, double *a, int i, int jlo, int jhi) {
    for (int j = jlo; j < jhi; ++j)
        AT(a, nj, i, j) = 10 * i + j;
}

void grid_init(int ni, int nj, double *a, double *b) {
    for (int i = 0; i < ni; ++i) {
        grid_init_span(nj, a, i, 0, nj);
        if (b)
            memset(&AT(b, nj, i, 0), 0, (size_t)nj * sizeof(double));
    }
}

int grid_write(const char *path, int ni, int nj, const double *x) {
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        return -1;
    }
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j)
            fprintf(f, "%f ", AT(x, nj, i, j));
        fprintf(f, "\n");
    }
    if (fclose(f) != 0) {
        perror(path);
        return -1;
    }
    return 0;
}

size_t grid_diff(int ni, int nj, const double *x, const double *y) {
    size_t n = (size_t)ni * (size_t)nj, diff = 0;
    for (size_t k = 0; k < n; ++k) {
        uint64_t u, v;
        memcpy(&u, &x[k], sizeof u);
        memcpy(&v, &y[k], sizeof v);
        diff += u != v;
    }
    return diff;
}

double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

size_t check_against_seq(const char *name, int ni, int nj,
                         const double *result) {
    const struct loop_variant *v = seq_find(name);
    if (!v) {
        fprintf(stderr, "check: no sequential variant '%s'\n", name);
        return 1;
    }
    double *a = grid_alloc(ni, nj), *b = v->uses_b ? grid_alloc(ni, nj) : NULL;
    grid_init(ni, nj, a, b);
    v->fn(ni, nj, a, b);
    size_t bad = grid_diff(ni, nj, v->uses_b ? b : a, result);
    if (bad)
        fprintf(stderr, "check FAIL %s %dx%d: %zu of %zu elements differ\n",
                name, ni, nj, bad, (size_t)ni * (size_t)nj);
    else
        fprintf(stderr, "check ok %s %dx%d\n", name, ni, nj);
    grid_free(a);
    grid_free(b);
    return bad;
}
