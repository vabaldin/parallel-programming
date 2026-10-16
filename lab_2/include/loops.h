#ifndef LOOPS_H
#define LOOPS_H

#include <math.h>
#include <stddef.h>

/* Grids are row-major ni x nj arrays of doubles: x[i][j] is x[i * nj + j]. */
#define AT(x, nj, i, j) ((x)[(size_t)(i) * (size_t)(nj) + (size_t)(j)])

double *grid_alloc(int ni, int nj);
void grid_free(double *x);

/* a[i][j] = 10 * i + j and, if b is not NULL, b[i][j] = 0. */
void grid_init(int ni, int nj, double *a, double *b);
/* The same for a[i][jlo..jhi) only. */
void grid_init_span(int nj, double *a, int i, int jlo, int jhi);

/* Writes the grid in the manual's format: "%f " per element, one row per
 * line. Returns 0 on success. */
int grid_write(const char *path, int ni, int nj, const double *x);

/* Number of elements whose bit patterns differ. */
size_t grid_diff(int ni, int nj, const double *x, const double *y);

double wtime(void);

/* The loop bodies are shared so that every implementation performs exactly
 * the same floating-point operations and the outputs match bit for bit. */
static inline void ref_row(double *a_i, int nj) {
    for (int j = 0; j < nj; ++j)
        a_i[j] = sin(2 * a_i[j]);
}

/* 1b: a[i][j] = sin(5 * a[i-2][j+3]) for j in [jlo, jhi). */
static inline void v1b_row(double *a_i, const double *a_im2, int jlo,
                           int jhi) {
    for (int j = jlo; j < jhi; ++j)
        a_i[j] = sin(5 * a_im2[j + 3]);
}

/* 2b: a[i][j] = sin(0.2 * a[i+1][j-6]) for j in [jlo, jhi). */
static inline void v2b_row(double *a_i, const double *a_ip1, int jlo,
                           int jhi) {
    for (int j = jlo; j < jhi; ++j)
        a_i[j] = sin(0.2 * a_ip1[j - 6]);
}

/* 3b, first loop: a[i][j] = sin(0.01 * a[i][j]). */
static inline void v3b_row_a(double *a_i, int nj) {
    for (int j = 0; j < nj; ++j)
        a_i[j] = sin(0.01 * a_i[j]);
}

/* 3b, second loop: b[i][j] = a[i+1][j-3] * 2. */
static inline void v3b_row_b(double *b_i, const double *a_ip1, int nj) {
    for (int j = 3; j < nj; ++j)
        b_i[j] = a_ip1[j - 3] * 2;
}

/* A loop nest from the manual. Variants that do not use b ignore it. */
typedef void (*loop_fn)(int ni, int nj, double *a, double *b);

struct loop_variant {
    const char *name;
    int uses_b;
    loop_fn fn;
};

/* Sequential reference implementations: ref, 1b, 2b, 3b. */
extern const struct loop_variant seq_variants[];
extern const size_t seq_variants_count;
const struct loop_variant *seq_find(const char *name);

void seq_ref(int ni, int nj, double *a, double *b);
void seq_1b(int ni, int nj, double *a, double *b);
void seq_2b(int ni, int nj, double *a, double *b);
void seq_3b(int ni, int nj, double *a, double *b);

struct options {
    const char *variant;
    int ni, nj;
    int reps;
    int threads;
    const char *out;
    int check;
    int header;
};

/* Parses -v variant -n ISIZE -m JSIZE -r reps -t threads -o file -c -H.
 * Returns 0 on success; prints usage and returns -1 otherwise. */
int options_parse(int argc, char **argv, struct options *o);

#define CSV_HEADER "variant,impl,isize,jsize,workers,reps,min_s,mean_s,comp_s"

/* Runs the sequential version of variant `name` on a fresh grid and compares
 * its result (a, or b for variants that use b) with `result`. Prints the
 * outcome to stderr and returns the number of mismatching elements. */
size_t check_against_seq(const char *name, int ni, int nj,
                         const double *result);

#endif
