/* MPI versions of ref and 1b. Run with mpirun -np P. Every rank allocates
 * the whole grid but initializes and touches only the part it computes, so
 * the other pages are never mapped; rank 0 initializes everything, collects
 * the result, prints the CSV line and writes or checks the output. min_s
 * covers computation plus collection on rank 0, comp_s only the computation
 * (both are the maximum over ranks). */
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loops.h"

static int rank, nprocs;

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p)
        MPI_Abort(MPI_COMM_WORLD, 1);
    return p;
}

static int block_lo(int n, int p) { return (int)((long long)n * p / nprocs); }

static void init_ref(int ni, int nj, double *a) {
    for (int i = block_lo(ni, rank); i < block_lo(ni, rank + 1); ++i)
        grid_init_span(nj, a, i, 0, nj);
}

/* ref: no dependences, contiguous blocks of rows, MPI_Gatherv to rank 0. */
static void mpi_ref(int ni, int nj, double *a, double *comp_s) {
    double t0 = MPI_Wtime();
    int lo = block_lo(ni, rank), hi = block_lo(ni, rank + 1);
    for (int i = lo; i < hi; ++i)
        ref_row(&AT(a, nj, i, 0), nj);
    *comp_s = MPI_Wtime() - t0;

    if (nprocs == 1)
        return;
    int *counts = xmalloc((size_t)nprocs * sizeof(int));
    int *displs = xmalloc((size_t)nprocs * sizeof(int));
    for (int p = 0; p < nprocs; ++p) {
        counts[p] = (block_lo(ni, p + 1) - block_lo(ni, p)) * nj;
        displs[p] = block_lo(ni, p) * nj;
    }
    if (rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, a, counts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(&AT(a, nj, lo, 0), counts[rank], MPI_DOUBLE, NULL, NULL,
                    NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(counts);
    free(displs);
}

/* 1b: a[i][j] = sin(5 * a[i-2][j+3]), distance vector (2, -3).
 * With the skew k = j + 3 * (i / 2), iteration (i, j) and its source
 * (i-2, j+3) have the same k, so every k is an independent chain and the
 * k axis can be split between ranks with no communication during the
 * computation. In each row a rank owns one contiguous run of columns. */
static int skew(int i) { return 3 * (i / 2); }

/* Splits the k range so that every rank gets about the same number of
 * computed elements. bounds has nprocs + 1 entries. */
static void split_1b(int ni, int nj, int *bounds) {
    int kmax = nj - 4 + skew(ni - 1);
    int nk = kmax + 2;
    long long *cnt = xmalloc((size_t)nk * sizeof(long long));
    memset(cnt, 0, (size_t)nk * sizeof(long long));
    for (int i = 2; i < ni; ++i) {
        cnt[skew(i)] += 1;
        cnt[nj - 3 + skew(i)] -= 1;
    }
    long long total = 0, run = 0;
    for (int k = 0; k < nk; ++k) {
        run += cnt[k];
        cnt[k] = run;
        total += run;
    }
    bounds[0] = 0;
    long long acc = 0;
    int k = 0;
    for (int p = 1; p < nprocs; ++p) {
        long long target = total * p / nprocs;
        while (k < nk && acc < target)
            acc += cnt[k++];
        bounds[p] = k;
    }
    bounds[nprocs] = nk;
    free(cnt);
}

/* Columns of row i in the chains of rank p, clipped to [0, jmax). */
static void cols_1b(int i, int jmax, const int *bounds, int p, int *jlo,
                    int *jhi) {
    int lo = bounds[p] - skew(i), hi = bounds[p + 1] - skew(i);
    *jlo = lo < 0 ? 0 : lo;
    *jhi = hi > jmax ? jmax : hi;
}

/* Every element of the rank's chains, including the initial values at their
 * ends in rows 0 and 1 and in the last three columns. */
static void init_1b(int ni, int nj, double *a) {
    if (ni <= 2 || nj <= 3)
        return;
    int *bounds = xmalloc((size_t)(nprocs + 1) * sizeof(int));
    split_1b(ni, nj, bounds);
    for (int i = 0; i < ni; ++i) {
        int jlo, jhi;
        cols_1b(i, nj, bounds, rank, &jlo, &jhi);
        grid_init_span(nj, a, i, jlo, jhi);
    }
    free(bounds);
}

/* The rows [2, ni) owned by rank p as an indexed datatype over the grid. */
static MPI_Datatype type_1b(int ni, int nj, const int *bounds, int p) {
    int *lens = xmalloc((size_t)ni * sizeof(int));
    int *disps = xmalloc((size_t)ni * sizeof(int));
    int n = 0;
    for (int i = 2; i < ni; ++i) {
        int jlo, jhi;
        cols_1b(i, nj - 3, bounds, p, &jlo, &jhi);
        if (jhi > jlo) {
            lens[n] = jhi - jlo;
            disps[n] = i * nj + jlo;
            ++n;
        }
    }
    MPI_Datatype t;
    MPI_Type_indexed(n, lens, disps, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    free(lens);
    free(disps);
    return t;
}

static void mpi_1b(int ni, int nj, double *a, double *comp_s) {
    double t0 = MPI_Wtime();
    if (ni <= 2 || nj <= 3) {
        *comp_s = 0;
        return;
    }
    int *bounds = xmalloc((size_t)(nprocs + 1) * sizeof(int));
    split_1b(ni, nj, bounds);
    for (int i = 2; i < ni; ++i) {
        int jlo, jhi;
        cols_1b(i, nj - 3, bounds, rank, &jlo, &jhi);
        v1b_row(&AT(a, nj, i, 0), &AT(a, nj, i - 2, 0), jlo, jhi);
    }
    *comp_s = MPI_Wtime() - t0;

    if (rank == 0) {
        MPI_Request *reqs = xmalloc((size_t)nprocs * sizeof(MPI_Request));
        MPI_Datatype *types = xmalloc((size_t)nprocs * sizeof(MPI_Datatype));
        for (int p = 1; p < nprocs; ++p) {
            types[p] = type_1b(ni, nj, bounds, p);
            MPI_Irecv(a, 1, types[p], p, 0, MPI_COMM_WORLD, &reqs[p]);
        }
        MPI_Waitall(nprocs - 1, reqs + 1, MPI_STATUSES_IGNORE);
        for (int p = 1; p < nprocs; ++p)
            MPI_Type_free(&types[p]);
        free(reqs);
        free(types);
    } else {
        MPI_Datatype t = type_1b(ni, nj, bounds, rank);
        MPI_Send(a, 1, t, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&t);
    }
    free(bounds);
}

static const struct mpi_variant {
    const char *name;
    void (*init)(int ni, int nj, double *a);
    void (*fn)(int ni, int nj, double *a, double *comp_s);
} variants[] = {
    {"ref", init_ref, mpi_ref},
    {"1b", init_1b, mpi_1b},
};
#define NVARIANTS (sizeof(variants) / sizeof(variants[0]))

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    struct options o;
    if (options_parse(argc, argv, &o) != 0)
        MPI_Abort(MPI_COMM_WORLD, 1);
    const struct mpi_variant *v = NULL;
    for (size_t i = 0; i < NVARIANTS; ++i)
        if (strcmp(variants[i].name, o.variant) == 0)
            v = &variants[i];
    if (!v) {
        if (rank == 0) {
            fprintf(stderr, "unknown variant '%s'; available:", o.variant);
            for (size_t i = 0; i < NVARIANTS; ++i)
                fprintf(stderr, " %s", variants[i].name);
            fprintf(stderr, "\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if ((long long)o.ni * o.nj > INT_MAX) {
        if (rank == 0)
            fprintf(stderr, "grid too large for int MPI counts\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *a = grid_alloc(o.ni, o.nj);
    double min_s = 0, total_s = 0, min_comp = 0;
    for (int r = 0; r < o.reps; ++r) {
        if (rank == 0)
            grid_init(o.ni, o.nj, a, NULL);
        else
            v->init(o.ni, o.nj, a);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime(), comp;
        v->fn(o.ni, o.nj, a, &comp);
        double t[2] = {MPI_Wtime() - t0, comp};
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : t, t, 2, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        total_s += t[0];
        if (r == 0 || t[0] < min_s)
            min_s = t[0];
        if (r == 0 || t[1] < min_comp)
            min_comp = t[1];
    }

    int status = 0;
    if (rank == 0) {
        if (o.header)
            printf("%s\n", CSV_HEADER);
        printf("%s,mpi,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", o.variant, o.ni, o.nj,
               nprocs, o.reps, min_s, total_s / o.reps, min_comp);
        fflush(stdout);
        if (o.out && grid_write(o.out, o.ni, o.nj, a) != 0)
            status = 1;
        if (o.check && check_against_seq(o.variant, o.ni, o.nj, a) != 0)
            status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    grid_free(a);
    MPI_Finalize();
    return status;
}
