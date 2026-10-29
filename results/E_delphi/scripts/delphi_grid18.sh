#!/bin/bash
set -u
export RAYON_NUM_THREADS=18
export OUT=/root/delphi/run/bench18
mkdir -p "$OUT"
BASE=8800
for L in 0 7; do
  BASE=$((BASE + 20))
  echo "=== 18-thread layers=$L ==="
  /root/delphi/run/delphi_bench.sh "$L" 3 "$BASE"
done
