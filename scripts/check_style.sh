#!/usr/bin/env bash
# Usage: scripts/check_style.sh [build-dir]
# Checks formatting (clang-format), Google style and naming (clang-tidy) and
# that text files are ASCII only. clang-tidy reads compile_commands.json from
# the build directory, so configure with CMake first.
set -euo pipefail

cd "$(dirname "$0")/.."
build=${1:-build}
status=0

git ls-files -z '*.cc' '*.h' | xargs -0 clang-format --dry-run --Werror ||
  status=1

# The compile commands do not always carry -isysroot (CMake leaves it out on
# some macOS setups), and clang-tidy then misses the SDK headers.
sysroot=()
if command -v xcrun >/dev/null 2>&1; then
  sysroot=(--extra-arg="-isysroot$(xcrun --show-sdk-path)")
fi
git ls-files -z '*.cc' |
  xargs -0 -n 1 -P "$(getconf _NPROCESSORS_ONLN)" \
    clang-tidy -p "$build" --quiet ${sysroot[@]+"${sysroot[@]}"} \
    2>/dev/null || status=1

if git ls-files -z -- ':!*.pdf' ':!*.png' |
  xargs -0 perl -ne 'if (/[^\x00-\x7F]/) { print "$ARGV:$.: non-ASCII\n"; $f = 1 }
                     close ARGV if eof; END { exit $f }'; then :; else status=1; fi

exit $status
