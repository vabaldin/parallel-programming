#!/bin/sh
# Usage: lab_2/scripts/bench.sh [variant ...]   (default: ref 1b 2b 3b)
# Times the sequential program and every parallel implementation of each
# variant for ISIZE = JSIZE = 1000 and 5000 and 1, 2, 4, 6, 8, 11 workers,
# writing lab_2/results/<variant>.csv. Set LOCK to a command prefix that
# serializes benchmark runs on a shared machine; each (variant, size) pair is
# one locked run. Run from the repository root after building into build/.
set -eu

BUILD=${BUILD:-build/lab_2}
OUT=${OUT:-lab_2/results}
LOCK=${LOCK:-}
WORKERS=${WORKERS:-"1 2 4 6 8 11"}
SIZES=${SIZES:-"1000 5000"}
MPIRUN=${MPIRUN:-mpirun}

reps_for() { if [ "$1" -ge 5000 ]; then echo 7; else echo 30; fi; }

impls_for() {
    case $1 in
    ref) echo "omp:ref mpi:ref" ;;
    1b) echo "mpi:1b" ;;
    2b) echo "omp:2b omp:2b-cols" ;;
    3b) echo "omp:3b" ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# One (variant, size) pair; prints CSV rows without the header.
run_size() {
    v=$1 n=$2 r=$(reps_for "$2")
    "$BUILD/lab2_seq" -v "$v" -n "$n" -m "$n" -r "$r"
    for impl in $(impls_for "$v"); do
        tech=${impl%%:*} name=${impl#*:}
        for p in $WORKERS; do
            if [ "$tech" = mpi ]; then
                "$MPIRUN" -np "$p" "$BUILD/lab2_mpi" -v "$name" -n "$n" -m "$n" -r "$r"
            else
                "$BUILD/lab2_omp" -v "$name" -n "$n" -m "$n" -r "$r" -t "$p"
            fi
        done
    done
}

if [ "${1:-}" = --run-size ]; then
    run_size "$2" "$3"
    exit 0
fi

mkdir -p "$OUT"
[ $# -gt 0 ] || set -- ref 1b 2b 3b
for v in "$@"; do
    impls_for "$v" >/dev/null
    csv="$OUT/$v.csv"
    echo "variant,impl,isize,jsize,workers,reps,min_s,mean_s,comp_s" > "$csv"
    for n in $SIZES; do
        echo "bench $v n=$n" >&2
        $LOCK "$0" --run-size "$v" "$n" >> "$csv"
    done
done
