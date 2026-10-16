/* Sequential reference programs. Usage: see options_parse. */
#include <stdio.h>

#include "loops.h"

int main(int argc, char **argv) {
    struct options o;
    if (options_parse(argc, argv, &o) != 0)
        return 1;
    const struct loop_variant *v = seq_find(o.variant);
    if (!v) {
        fprintf(stderr, "unknown variant '%s'; available:", o.variant);
        for (size_t i = 0; i < seq_variants_count; ++i)
            fprintf(stderr, " %s", seq_variants[i].name);
        fprintf(stderr, "\n");
        return 1;
    }

    double *a = grid_alloc(o.ni, o.nj);
    double *b = v->uses_b ? grid_alloc(o.ni, o.nj) : NULL;
    double min_s = 0, total_s = 0;
    for (int r = 0; r < o.reps; ++r) {
        grid_init(o.ni, o.nj, a, b);
        double t0 = wtime();
        v->fn(o.ni, o.nj, a, b);
        double dt = wtime() - t0;
        total_s += dt;
        if (r == 0 || dt < min_s)
            min_s = dt;
    }

    if (o.header)
        printf("%s\n", CSV_HEADER);
    printf("%s,seq,%d,%d,1,%d,%.6f,%.6f,%.6f\n", v->name, o.ni, o.nj, o.reps,
           min_s, total_s / o.reps, min_s);

    int status = 0;
    if (o.out && grid_write(o.out, o.ni, o.nj, v->uses_b ? b : a) != 0)
        status = 1;
    grid_free(a);
    grid_free(b);
    return status;
}
