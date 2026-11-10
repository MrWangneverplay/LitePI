#!/usr/bin/env bash
# 本地 loopback 的 r_h 端到端测量：起 mnist 服务端 -> netem(lo) 客户端 -> 收尾清理。
# 用法：RTT=20,50,100 BW=50 REPEAT=5 bash scripts/run_e3_rh_local.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$PWD/.venv311/bin/python3.11"
PORT="${PORT:-9101}"
RTT="${RTT:-20,50,100}"
BW="${BW:-50}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-3}"
SERVER_SCHEDULE="${SERVER_SCHEDULE:-1,1}"
SCHEDULE="${SCHEDULE:-11}"

sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0
"$PY" -B analysis/pi_server.py --dataset mnist --schedule "$SERVER_SCHEDULE" --serve \
    --port "$PORT" > /tmp/pi_server_e3.log 2>&1 &
SRV=$!
cleanup() {
    kill "$SRV" 2>/dev/null || true
    sudo tc qdisc del dev lo root 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 60); do
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then break; fi
    sleep 2
done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "服务端未在 120s 内监听 $PORT" >&2
    tail -20 /tmp/pi_server_e3.log >&2 || true
    exit 1
fi
echo "服务端已监听 $PORT"

sudo -E "$PY" -B analysis/netem_e2e.py \
    --host 127.0.0.1 --port "$PORT" --iface lo --device desktop \
    --dataset mnist --schedule "$SCHEDULE" --mode e3 \
    --rtts "$RTT" --bws "$BW" --repeat "$REPEAT" --warmup "$WARMUP"
