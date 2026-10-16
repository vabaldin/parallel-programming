#include "matmul.h"

void matmul_transposed(size_t n, const float *a, const float *b, float *c) {
    float *bt = matrix_alloc(n);
    matrix_transpose(n, b, bt);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k)
                sum += a[i * n + k] * bt[j * n + k];
            c[i * n + j] = sum;
        }
    }
    matrix_free(bt);
}
