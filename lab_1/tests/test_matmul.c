/* Usage: test_matmul [variant ...] */
#include <stdio.h>
#include <string.h>

#include "matmul.h"

#define REL_TOL 1e-3f

static const size_t sizes[] = {1, 2, 7, 16, 64, 100, 128, 256, 300, 513};
#define NSIZES (sizeof(sizes) / sizeof(sizes[0]))

static int check(const struct matmul_variant *v) {
    const struct matmul_variant *ref = &matmul_variants[0];
    int failures = 0;

    for (size_t s = 0; s < NSIZES; ++s) {
        size_t n = sizes[s];
        float *a = matrix_alloc(n), *b = matrix_alloc(n);
        float *expected = matrix_alloc(n), *actual = matrix_alloc(n);

        matrix_fill_value(n, a, 1.0f);
        matrix_fill_value(n, b, 1.0f);
        matrix_fill_value(n, expected, (float)n);
        matrix_fill_value(n, actual, -1.0f);
        v->fn(n, a, b, actual);
        size_t bad = matrix_compare(n, expected, actual, 0.0f);
        if (bad) {
            printf("FAIL %s n=%zu ones: %zu mismatches\n", v->name, n, bad);
            ++failures;
        }

        matrix_fill_random(n, a, (unsigned)(2 * n + 1));
        matrix_fill_random(n, b, (unsigned)(2 * n + 2));
        ref->fn(n, a, b, expected);
        matrix_fill_value(n, actual, -1.0f);
        v->fn(n, a, b, actual);
        bad = matrix_compare(n, expected, actual, REL_TOL);
        if (bad) {
            printf("FAIL %s n=%zu random: %zu mismatches\n", v->name, n, bad);
            ++failures;
        }

        matrix_free(a);
        matrix_free(b);
        matrix_free(expected);
        matrix_free(actual);
    }

    printf("%s %s\n", failures ? "FAIL" : "ok  ", v->name);
    return failures;
}

int main(int argc, char **argv) {
    int failures = 0;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            const struct matmul_variant *v = matmul_find(argv[i]);
            if (!v) {
                printf("unknown variant '%s'\n", argv[i]);
                return 1;
            }
            failures += check(v);
        }
    } else {
        for (size_t i = 0; i < matmul_variants_count; ++i)
            failures += check(&matmul_variants[i]);
    }
    return failures ? 1 : 0;
}
