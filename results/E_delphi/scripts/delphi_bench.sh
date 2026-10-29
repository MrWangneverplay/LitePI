#!/bin/bash
set -u
BIN=/root/delphi/rust/target/release
export LD_LIBRARY_PATH=/root/libtorch/lib
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
OUT=${OUT:-/root/delphi/run/bench}
mkdir -p "$OUT"

LAYERS="${1:-7}"
RUNS="${2:-5}"
BASE_PORT="${3:-8100}"

for r in $(seq 1 "$RUNS"); do
  port=$((BASE_PORT + r))
  slog="$OUT/server_L${LAYERS}_r${r}.log"
  clog="$OUT/client_L${LAYERS}_r${r}.log"
  /usr/bin/time -v "$BIN/minionn-server" -l "$LAYERS" -p "$port" > "$slog" 2> "$OUT/server_L${LAYERS}_r${r}.time" &
  spid=$!
  sleep 0.7
  /usr/bin/time -v "$BIN/minionn-client" -i 127.0.0.1 -l "$LAYERS" -p "$port" > "$clog" 2> "$OUT/client_L${LAYERS}_r${r}.time"
  cexit=$?
  wait $spid
  sexit=$?
  echo "RESULT layers=$LAYERS run=$r port=$port client_exit=$cexit server_exit=$sexit"
  sleep 1
done
