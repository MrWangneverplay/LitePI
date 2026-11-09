#!/usr/bin/env bash
# 60 组 Hybrid schema2 评测，默认 maskv3、8 并发，最多 8 并发。
# WZY_DIR/ROOT 指定数据与结果根目录；代码位置始终由本脚本定位。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)" || exit 1
PYTHON="${PYTHON:-python}"
WZY_DIR="${WZY_DIR:-$REPO_DIR}"
ROOT="${ROOT:-$WZY_DIR}"
ROOT="$(cd -- "$ROOT" && pwd)" || exit 1
SAMPLES="${SAMPLES:-64}"
JOBS="${JOBS:-8}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
BACKEND="${BACKEND:-seal}"
WIRE="${WIRE:-baseline}"
MASK_EXACT="${MASK_EXACT:-1}"

if [[ ! "$JOBS" =~ ^[1-8]$ ]]; then
  printf 'JOBS 必须是 1 至 8 的整数\n' >&2
  exit 2
fi
if [[ ! "$SAMPLES" =~ ^[0-9]+$ || ! "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
  printf 'SAMPLES 与 SAMPLE_SEED 必须是非负整数\n' >&2
  exit 2
fi
case "$BACKEND" in
  seal|tenseal|stub) ;;
  *) printf 'BACKEND 必须是 seal、tenseal 或 stub\n' >&2; exit 2 ;;
esac
if [[ ! "$WIRE" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'WIRE 名称无效\n' >&2
  exit 2
fi
case "$MASK_EXACT" in
  0|1) ;;
  *) printf 'MASK_EXACT 必须是 0 或 1\n' >&2; exit 2 ;;
esac
command -v "$PYTHON" >/dev/null || exit 1

# 顺序与 eval_hybrid_accuracy.py 的默认 out-tag 完全一致。
SUFFIX="_${BACKEND}"
if [[ "$WIRE" != baseline ]]; then
  SUFFIX+="_wire${WIRE}"
fi
if [[ "$MASK_EXACT" == 1 ]]; then
  SUFFIX+="_maskv3"
fi
SUFFIX+="_v2"
LOG_ROOT="$ROOT/logs/hybrid_accuracy${SUFFIX}"
mkdir -p -- "$LOG_ROOT" || exit 1
LOG_DIR="$(mktemp -d "$LOG_ROOT/run_XXXXXXXX")" || exit 1
printf '本批配置: samples=%s sample_seed=%s jobs=%s backend=%s wire=%s mask_exact=%s\n' \
  "$SAMPLES" "$SAMPLE_SEED" "$JOBS" "$BACKEND" "$WIRE" "$MASK_EXACT"
printf '仅使用本批 v2 后缀: %s；日志: %s\n' "$SUFFIX" "$LOG_DIR"

run_one() {
  local ds="$1" tag="$2" sd="$3" rc
  local stem="${1}_s${2}_seed${3}"
  local record="$ROOT/results/raw_results/E1/$ds/hybrid_s${tag}_acc${SUFFIX}/seed${sd}.json"
  local log="$LOG_DIR/${stem}.log"
  local -a cmd=("$PYTHON" "$REPO_DIR/analysis/eval_hybrid_accuracy.py"
    --root "$ROOT" --dataset "$ds" --schedule "$tag" --seed "$sd"
    --samples "$SAMPLES" --sample-seed "$SAMPLE_SEED"
    --backend "$BACKEND" --wire "$WIRE")
  if [[ "$MASK_EXACT" == 1 ]]; then
    cmd+=(--mask-exact)
  fi
  if [[ -f "$record" ]]; then
    cmd+=(--resume)
  fi
  {
    printf '本批目标记录: %s\n' "$record"
    printf '命令: '; printf '%q ' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
  } > "$log" 2>&1
  rc=$?
  printf 'eval_rc=%s\n' "$rc" >> "$log" || return 1
  printf '%s\n' "$rc" > "$LOG_DIR/${stem}.status" || return 1
  printf 'rc=%s %s s=%s seed=%s record=%s log=%s\n' "$rc" "$ds" "$tag" "$sd" "$record" "$log"
  return "$rc"
}
export PYTHON REPO_DIR ROOT SAMPLES SAMPLE_SEED BACKEND WIRE MASK_EXACT SUFFIX LOG_DIR
export -f run_one

generate_tasks() {
  local ds tag sd
  for ds in mnist cifar10 uci_har; do
    for tag in 00 01 10 11; do
      for sd in 73 74 75 76 77; do
        printf '%s %s %s\n' "$ds" "$tag" "$sd"
      done
    done
  done
}

batch_rc=0
generate_tasks | xargs -P "$JOBS" -n 3 bash -c 'run_one "$@"' _ || batch_rc=$?

success=0
failed=0
unreported=0
records=0
for ds in mnist cifar10 uci_har; do
  for tag in 00 01 10 11; do
    for sd in 73 74 75 76 77; do
      status="$LOG_DIR/${ds}_s${tag}_seed${sd}.status"
      if [[ -f "$status" ]] && read -r rc < "$status"; then
        if [[ "$rc" == 0 ]]; then
          ((success += 1))
        else
          ((failed += 1))
        fi
      else
        ((unreported += 1))
      fi
      record="$ROOT/results/raw_results/E1/$ds/hybrid_s${tag}_acc${SUFFIX}/seed${sd}.json"
      if [[ -f "$record" ]]; then
        ((records += 1))
      fi
    done
  done
done
printf '本批调用统计: 成功=%s 失败=%s 未回报=%s / 60\n' "$success" "$failed" "$unreported"
printf '仅本批 v2 目标目录存在记录=%s / 60（可能含 partial，不代表完整或有效）\n' "$records"
if (( batch_rc != 0 )); then
  printf '批量调用失败，xargs/管道退出码=%s；请检查本批日志\n' "$batch_rc" >&2
  exit "$batch_rc"
fi
if (( failed != 0 || unreported != 0 )); then
  printf '存在失败或未回报任务，不宣称全部完成\n' >&2
  exit 1
fi
exit 0
