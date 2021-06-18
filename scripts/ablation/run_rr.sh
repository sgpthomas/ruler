#!/usr/bin/env bash

# determine physical directory of this script
# h/t Zach
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"

TIMESTAMP="$(date "+%Y-%m-%d_%H%M")"

"$MYDIR/run_ruler_rr.sh" -d bv4 -o "$MYDIR/output/$TIMESTAMP" -r 10
"$MYDIR/run_ruler_rr.sh" -d bv32 -o "$MYDIR/output/$TIMESTAMP" -r 10
"$MYDIR/run_ruler_rr.sh" -d rational -o "$MYDIR/output/$TIMESTAMP" -r 10  --use-smt
node parse.js "$MYDIR/output/$TIMESTAMP/"
python visualize.py