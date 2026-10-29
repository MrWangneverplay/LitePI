#!/bin/bash
set -u
BIN=/root/delphi/rust/target/release
export LD_LIBRARY_PATH=/root/libtorch/lib
export RAYON_NUM_THREADS=4
OUT=/root/delphi/run/mnist_bench
mkdir -p "$OUT"
port=8600
for L in 3 0; do
  runs=3
  [ "$L" = "0" ] && runs=1
  for r in $(seq 1 $runs); do
    port=$((port + 1))
    /usr/bin/time -v "$BIN/mnist-server" -l "$L" -p "$port" > "$OUT/server_L${L}_r${r}.log" 2> "$OUT/server_L${L}_r${r}.time" &
    spid=$!
    sleep 0.7
    /usr/bin/time -v "$BIN/mnist-client" -i 127.0.0.1 -l "$L" -p "$port" > "$OUT/client_L${L}_r${r}.log" 2> "$OUT/client_L${L}_r${r}.time"
    cexit=$?
    wait $spid
    echo "MNIST layers=$L run=$r client_exit=$cexit"
    sleep 1
  done
done
