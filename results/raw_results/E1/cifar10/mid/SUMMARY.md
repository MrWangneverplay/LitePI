# E1 CIFAR-10 模型 B（conv3）精度与密文实测汇总

更新时间：2026-11-02 15:40 CST（远程扫描仍在进行，完成后需要重新生成本表）

## 1. 明文基线（3 种子：73 / 74 / 75，50 epoch）

| 调度 s | 非线性层 ReLU 数 K_h | 种子 73 / 74 / 75（%） | 均值 ± 样本标准差（%） | export_logit_err |
|---|---|---|---|---|
| 0000 | 0 | 79.03 / 78.81 / 78.22 | 78.69 ± 0.42 | 4.1e-6 ~ 5.3e-6 |
| 0100 | 1 | 82.91 / 83.39 / 82.90 | 83.07 ± 0.28 | 5.1e-6 ~ 6.4e-6 |
| 0110 | 2 | 83.61 / 83.74 / 83.41 | 83.59 ± 0.17 | 4.5e-6 ~ 5.0e-6 |
| 1110 | 2 | 84.13 / 84.08 / 83.97 | 84.06 ± 0.08 | 4.3e-6 ~ 5.4e-6 |
| 1111 | 3 | 82.89 / 83.17 / 83.36 | 83.14 ± 0.24 | 3.3e-6 ~ 3.8e-6 |

来源：`s<sched>/seed<n>.json`（每个 260 B，只含 test_acc / export_logit_err / 训练元数据）。

## 2. 密文实测（已完成的整组）

条件：SEAL 4.1.2，N=8192，8 线程，`--mask-exact`（M2 v3），wire=baseline，64 个测试样本，
sample_seed=20261030。

| 调度 | 种子 | agreement | accuracy（64 样本） | wall_s 均值（min–max） | 密文-明文 max\|logit err\| 均值（min–max） | 服务端 forward / linear / hybrid-ReLU / MPC (s) |
|---|---|---|---|---|---|---|
| 0100 | 73 | 1.0000 | 0.84375 | 21.77（21.66–21.96） | 4.31e-2（1.59e-2–1.298e-1） | 21.17 / 21.09 / 0.086 / 0.022 |

字节口径（同一组 64 次查询逐次相同）：

| 计数位置 | 上行 | 下行 | 说明 |
|---|---|---|---|
| 客户端 `online_up_bytes` | 12,819,868 B | 1,204,193 B | 不含密钥；含每次新会话上传的 context（`context_upload_s`≈0.42 s，占 wall 1.9%） |
| 服务端 `bytes_received / bytes_sent`（单连接） | 196,237,780 B | 786,708 B | 上行含一次性密钥 184.6 MB + context；下行只含协议消息 |

结论：**当前 `up_bytes` 不能作为“每查询通信量”**。`eval_deep_accuracy.py` 每个样本新建一次客户端会话
（server log 中 64 条记录的 `query_index` 全为 0、`reused_context=false`），因此密钥与 context 被重复计入上行。
论文表 7-11 的 1.18 / 5.11 MB 是 canonical 模型在持久会话下的口径，模型 B 行必须同样以持久会话重测才能并列。

## 3. 不可部署调度（s=0000）

三个种子都在**第 1 个样本**即失败，日志在 `_infeasible_s0000/seed73..75.log`：

```
pi_proto.ProtocolError: ValueError: scale out of bounds
```

原因：全 P2 需要 9 层乘法深度，paper profile 预算 6 层。与论文现有表述“深度超预算，不可部署”一致，
可直接作为附录证据；不建议为它保留远程算力。

## 4. 当前远程状态（2026-11-02 15:38 CST）

用户侧已启动的全调度扫描脚本（bash 循环，`eval_deep_accuracy.py --samples 64 --threads 8 --mask-exact`）正在运行：
`s0100 seed74` 已到 18/64，单样本约 21.7 s。按此推算，剩余 10 组（s0100 seed75、s0110 ×3、s1110 ×3、s1111 ×3）
还需约 3.5–4 小时。本文档表格待扫描结束后补齐。

## 5. 复现命令（远程，由用户在 FinalShell 执行）

```bash
cd ~/wzy
python3 analysis/eval_deep_accuracy.py --schedule 0100 --seed 74 --samples 64 --threads 8 --mask-exact
```

汇总脚本（仅在本机读取 JSON，不触发计算）：

```bash
python - <<'PY'
import json, glob, statistics
for p in sorted(glob.glob('results/raw_results/E1/cifar10/mid/s*_cipher*/seed*.json')):
    d = json.load(open(p))
    w = [r['wall_s'] for r in d['rows']]; e = [r['max_abs_err'] for r in d['rows']]
    print(p, d['agreement'], d['accuracy'], round(statistics.mean(w), 2),
          '%.2e' % statistics.mean(e), d['threads'], d['mask_exact'])
PY
```
