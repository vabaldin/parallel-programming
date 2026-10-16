#include <string.h>

#include "matmul.h"

const struct matmul_variant matmul_variants[] = {
    {"naive", matmul_naive},
    {"transposed", matmul_transposed},
    {"blocked", matmul_blocked},
    {"strassen", matmul_strassen},
};

const size_t matmul_variants_count =
    sizeof(matmul_variants) / sizeof(matmul_variants[0]);

const struct matmul_variant *matmul_find(const char *name) {
    for (size_t i = 0; i < matmul_variants_count; ++i)
        if (strcmp(matmul_variants[i].name, name) == 0)
            return &matmul_variants[i];
    return NULL;
}
