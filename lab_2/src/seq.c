#include <string.h>

#include "loops.h"

void seq_ref(int ni, int nj, double *a, double *b) {
    (void)b;
    for (int i = 0; i < ni; ++i)
        ref_row(&AT(a, nj, i, 0), nj);
}

void seq_1b(int ni, int nj, double *a, double *b) {
    (void)b;
    for (int i = 2; i < ni; ++i)
        v1b_row(&AT(a, nj, i, 0), &AT(a, nj, i - 2, 0), 0, nj - 3);
}

void seq_2b(int ni, int nj, double *a, double *b) {
    (void)b;
    for (int i = 0; i < ni - 1; ++i)
        v2b_row(&AT(a, nj, i, 0), &AT(a, nj, i + 1, 0), 6, nj);
}

void seq_3b(int ni, int nj, double *a, double *b) {
    for (int i = 0; i < ni; ++i)
        v3b_row_a(&AT(a, nj, i, 0), nj);
    for (int i = 0; i < ni - 1; ++i)
        v3b_row_b(&AT(b, nj, i, 0), &AT(a, nj, i + 1, 0), nj);
}

const struct loop_variant seq_variants[] = {
    {"ref", 0, seq_ref},
    {"1b", 0, seq_1b},
    {"2b", 0, seq_2b},
    {"3b", 1, seq_3b},
};
const size_t seq_variants_count = sizeof(seq_variants) / sizeof(seq_variants[0]);

const struct loop_variant *seq_find(const char *name) {
    for (size_t i = 0; i < seq_variants_count; ++i)
        if (strcmp(seq_variants[i].name, name) == 0)
            return &seq_variants[i];
    return NULL;
}
