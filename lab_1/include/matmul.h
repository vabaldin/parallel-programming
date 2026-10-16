#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/* All matrices are square n x n, row-major, contiguous floats.
 * A variant computes C = A * B. It must not modify A or B and must fully
 * overwrite C. Any scratch (transposed copies, padding, Strassen temporaries)
 * is allocated inside the variant so that its cost is part of the timing. */
typedef void (*matmul_fn)(size_t n, const float *a, const float *b, float *c);

struct matmul_variant {
    const char *name;
    matmul_fn fn;
};

/* Registry of all implemented variants; the first entry is the reference. */
extern const struct matmul_variant matmul_variants[];
extern const size_t matmul_variants_count;

const struct matmul_variant *matmul_find(const char *name);

void matmul_naive(size_t n, const float *a, const float *b, float *c);

/* Matrix helpers (src/matrix.c). Allocations are MATRIX_ALIGN-aligned. */
#define MATRIX_ALIGN 64

float *matrix_alloc(size_t n);
void matrix_free(float *m);
void matrix_fill_value(size_t n, float *m, float value);
void matrix_fill_random(size_t n, float *m, unsigned seed);
void matrix_transpose(size_t n, const float *src, float *dst);

/* Returns the number of elements where |expected - actual| exceeds
 * rel_tol * max(1, |expected|). */
size_t matrix_compare(size_t n, const float *expected, const float *actual,
                      float rel_tol);

#endif
