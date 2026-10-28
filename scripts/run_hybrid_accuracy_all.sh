#!/usr/bin/env bash
# 60 组（3 数据集 × 4 调度 × 5 种子）Hybrid 多样本精度评测，默认 8 并发。
# 用法：SAMPLES=64 JOBS=8 bash scripts/run_hybrid_accuracy_all.sh
set -u
cd /workspace/data/LitePI || exit 1

SAMPLES="${SAMPLES:-64}"
JOBS="${JOBS:-8}"
mkdir -p logs

run_one() {
  ds="$1"; tag="$2"; sd="$3"
  log="logs/hybacc_${ds}_s${tag}_seed${sd}.log"
  python analysis/eval_hybrid_accuracy.py \
    --dataset "$ds" --schedule "$tag" --seed "$sd" \
    --samples "$SAMPLES" --backend seal --resume > "$log" 2>&1
  rc=$?
  echo "rc=$rc ${ds} s=${tag} seed=${sd}"
}
export SAMPLES
export -f run_one

: > /tmp/hybacc_tasks.txt
for ds in mnist cifar10 uci_har; do
  for tag in 00 01 10 11; do
    for sd in 73 74 75 76 77; do
      echo "$ds $tag $sd" >> /tmp/hybacc_tasks.txt
    done
  done
done

xargs -a /tmp/hybacc_tasks.txt -P "$JOBS" -n 3 bash -c 'run_one "$@"' _

echo "=========================================="
echo "done: samples=$SAMPLES jobs=$JOBS"
ls results/raw_results/E1/*/hybrid_s*_acc_seal/seed*.json 2>/dev/null | wc -l
