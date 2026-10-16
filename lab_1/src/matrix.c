#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "matmul.h"

float *matrix_alloc(size_t n) {
    size_t bytes = n * n * sizeof(float);
    bytes = (bytes + MATRIX_ALIGN - 1) / MATRIX_ALIGN * MATRIX_ALIGN;
    return aligned_alloc(MATRIX_ALIGN, bytes ? bytes : MATRIX_ALIGN);
}

void matrix_free(float *m) { free(m); }

void matrix_fill_value(size_t n, float *m, float value) {
    for (size_t i = 0; i < n * n; ++i)
        m[i] = value;
}

/* Values in [0, 1) so that sums do not cancel and relative error stays
 * meaningful. */
void matrix_fill_random(size_t n, float *m, unsigned seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n * n; ++i) {
        state = state * 1664525u + 1013904223u;
        m[i] = (float)(state >> 8) / (float)(1u << 24);
    }
}

void matrix_transpose(size_t n, const float *src, float *dst) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            dst[j * n + i] = src[i * n + j];
}

size_t matrix_compare(size_t n, const float *expected, const float *actual,
                      float rel_tol) {
    size_t mismatches = 0;
    for (size_t i = 0; i < n * n; ++i) {
        float diff = fabsf(expected[i] - actual[i]);
        if (!(diff <= rel_tol * fmaxf(1.0f, fabsf(expected[i]))))
            ++mismatches;
    }
    return mismatches;
}
