#!/usr/bin/env bash
# api_decls.sh HEADER - the VI_API declarations of a vecindex.h, one a line, whitespace squeezed, comments dropped,
# sorted: what tests/compat/api-VERSION.txt records for a release and ./run checks every later header still declares.
set -euo pipefail
sed -e 's|/\*.*\*/||g' "$1" | awk '
    /\/\*/ { c = 1 } c { if (/\*\//) c = 0; next }
    /^VI_API/ { d = $0; while (d !~ /;/ && (getline l) > 0) d = d " " l; print d }' \
  | sed -e 's|/\*.*||' | tr -s ' \t' ' ' | sed -e 's/ *; */;/' -e 's/ *$//' | sort -u
