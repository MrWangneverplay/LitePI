#!/bin/bash
set -u
export RAYON_NUM_THREADS=4
export OUT=/root/delphi/run/bench
BASE=8300
for L in 0 1 2 3 5 6 7; do
  BASE=$((BASE + 20))
  echo "=== layers=$L base=$BASE ==="
  /root/delphi/run/delphi_bench.sh "$L" 5 "$BASE"
done
