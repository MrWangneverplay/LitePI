#!/usr/bin/env bash
# 批量导出 M2 mask-domain JSON：3 数据集 x 4 调度 x 5 种子 = 60 个。
# 用法：在项目根目录执行  bash scripts/gen_mask_domains_all.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

DATASETS="mnist cifar10 uci_har"
SCHEDULES="00 01 10 11"
SEEDS="73 74 75 76 77"

mkdir -p datas/mask_domains
ok=0; skip=0; fail=0

for ds in $DATASETS; do
  for tag in $SCHEDULES; do
    for sd in $SEEDS; do
      out="datas/mask_domains/${ds}_s${tag}_seed${sd}.json"

      if [ -f "$out" ]; then
        echo "[skip] $out 已存在"
        skip=$((skip+1)); continue
      fi

      # checkpoint 命名规则见 analysis/canonical_net.py:choose_checkpoint
      ckpt="datas/${ds}_hybrid_s${tag}_seed${sd}_cnn_train.pt"
      fb="datas/${ds}_hybrid_s${tag}_cnn_train.pt"
      if [ ! -f "$ckpt" ] && [ ! -f "$fb" ]; then
        echo "[miss] 无 checkpoint: $ckpt (或 ${fb##*/})  跳过 ${ds}/${tag}/seed${sd}"
        skip=$((skip+1)); continue
      fi

      echo "[gen ] $ds s=$tag seed=$sd -> $out"
      if python analysis/export_mask_domains.py \
           --dataset "$ds" --schedule "$tag" --seed "$sd" --out "$out"; then
        ok=$((ok+1))
      else
        echo "[FAIL] $ds s=$tag seed=$sd"
        fail=$((fail+1))
      fi
    done
  done
done

echo "=========================================="
echo "M2 domains: 生成 $ok, 跳过 $skip, 失败 $fail"
echo "已存在文件数: $(ls datas/mask_domains/*.json 2>/dev/null | wc -l) / 60"
