#!/usr/bin/env bash
# Hybrid 密文评测（SCI/CrypTFlow2 后端，r_h=1）：3 数据集 x 4 调度 x 5 种子 = 60 组。
# 用法：在项目根目录执行  bash scripts/run_hybrid_loopback_5seeds.sh
#
# 前置：
#   1) sci/litepi_sci.cpython-3XX-x86_64-linux-gnu.so 已就位
#   2) datas/mask_domains/*.json 已由 gen_mask_domains_all.sh 生成
#   3) 对应 checkpoint 存在
#
# 特点：
#   - 断点续跑：结果已存在且 mpc_backend 已是 litepi_sci 则跳过
#   - 旧的 python-OT 结果自动归档为 *.python_ot.json，不静默覆盖
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

DATASETS="mnist cifar10 uci_har"
SCHEDULES="00 01 10 11"
SEEDS="73 74 75 76 77"

ok=0; skip=0; fail=0

for ds in $DATASETS; do
  for tag in $SCHEDULES; do
    for sd in $SEEDS; do
      outdir="results/raw_results/E1/${ds}/hybrid_s${tag}_seal"
      out="${outdir}/seed${sd}.json"
      mkdir -p "$outdir"

      # 已有合规结果 -> 跳过
      if [ -f "$out" ]; then
        if grep -q '"mpc_runtime": *"litepi_sci"' "$out" 2>/dev/null; then
          echo "[skip] $out 已是 SCI 结果"
          skip=$((skip+1)); continue
        else
          if [ ! -f "${out%.json}.python_ot.json" ]; then
            mv "$out" "${out%.json}.python_ot.json"
            echo "[arch] 旧 python-OT 结果归档 -> ${out%.json}.python_ot.json"
          else
            echo "[arch] 归档已存在，丢弃当前 $out"
            rm -f "$out"
          fi
        fi
      fi

      # checkpoint 检查
      ckpt="datas/${ds}_hybrid_s${tag}_seed${sd}_cnn_train.pt"
      fb="datas/${ds}_hybrid_s${tag}_cnn_train.pt"
      if [ ! -f "$ckpt" ] && [ ! -f "$fb" ]; then
        echo "[miss] 无 checkpoint: $ckpt  跳过 ${ds}/${tag}/seed${sd}"
        skip=$((skip+1)); continue
      fi

      # 有 hybrid 层则必须有 M2 domains
      sched_arg="$(echo "$tag" | sed 's/./&,/g; s/,$//')"   # 11 -> 1,1
      md_arg=""
      if [ "$tag" != "00" ]; then
        md="datas/mask_domains/${ds}_s${tag}_seed${sd}.json"
        if [ ! -f "$md" ]; then
          echo "[miss] 无 M2 domains: $md  跳过 ${ds}/${tag}/seed${sd}"
          skip=$((skip+1)); continue
        fi
        md_arg="--mask-domains $md"
      fi

      echo "[run ] $ds s=$tag seed=$sd  (schedule=$sched_arg)"
      # shellcheck disable=SC2086
      if python analysis/pi_client.py \
           --loopback --backend seal \
           --dataset "$ds" --schedule "$sched_arg" --seed "$sd" \
           $md_arg --out "$out" --timeout 1800; then
        if grep -q '"mpc_runtime": *"litepi_sci"' "$out" 2>/dev/null; then
          ok=$((ok+1)); echo "       -> OK (SCI)"
        else
          echo "       -> 完成但 mpc_runtime 非 SCI！检查 sci/*.so"
          fail=$((fail+1))
        fi
      else
        echo "[FAIL] $ds s=$tag seed=$sd"
        fail=$((fail+1))
      fi
    done
  done
done

echo "=========================================="
echo "Hybrid 评测: 成功 $ok, 跳过 $skip, 失败 $fail"
echo "SCI 结果文件数: $(grep -l '"mpc_runtime": *"litepi_sci"' results/raw_results/E1/*/hybrid_s*_seal/*.json 2>/dev/null | wc -l)"
