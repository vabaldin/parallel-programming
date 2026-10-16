#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "loops.h"

static int positive(const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    return (*s && !*end && v > 0 && v <= 1000000) ? (int)v : -1;
}

int options_parse(int argc, char **argv, struct options *o) {
    *o = (struct options){.variant = "ref", .ni = 5000, .nj = 5000,
                          .reps = 3, .threads = 0};
    int opt;
    while ((opt = getopt(argc, argv, "v:n:m:r:t:o:cH")) != -1) {
        switch (opt) {
        case 'v': o->variant = optarg; break;
        case 'n': o->ni = positive(optarg); break;
        case 'm': o->nj = positive(optarg); break;
        case 'r': o->reps = positive(optarg); break;
        case 't': o->threads = positive(optarg); break;
        case 'o': o->out = optarg; break;
        case 'c': o->check = 1; break;
        case 'H': o->header = 1; break;
        default: goto usage;
        }
    }
    if (optind == argc && o->ni > 0 && o->nj > 0 && o->reps > 0 &&
        o->threads >= 0)
        return 0;
usage:
    fprintf(stderr,
            "usage: %s [-v variant] [-n ISIZE] [-m JSIZE] [-r reps]"
            " [-t threads] [-o file] [-c] [-H]\n",
            argv[0]);
    return -1;
}
