# Delphi 官方复现基线数据（E_delphi）

> 目的：为实验方案 §5 / Phase 3 提供 **Delphi 主基线（红线）** 的可复现实测数据。
> 官方仓库代码，仅含构建与 lint 兼容改动（密码学逻辑、网络结构、实验参数均未改，见 [CHANGES.md](./CHANGES.md)）。

## 1. 复现结论

- 官方 Delphi 实现（commit `92bc0071`）已在本机 **完整跑通**：编译 → CIFAR-10 MiniONN 密文推理 → 官方离线/在线双阶段延迟与通信量基准 → 官方 MNIST 示例。
- 官方 CIFAR-10 预训练 MiniONN（`7_approx`，7 个多项式激活层）在本机复现的密文预测与明文预测 **逐样本一致**（5/5 spot check），证明权重解析、输入布局、量化与 HE 参数链路正确。
- 官方声明的模型精度（`model_state` 记录，Keras，10k 测试集）为 **82.95%**；本机用同一组权重做明文复算，在固定种子的 100 张测试图上得到 **82.0%（82/100）**，与官方一致。

## 2. 测试环境

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 5 125H，18 逻辑核 |
| 内存 | 15 GB |
| OS | WSL2 Ubuntu（Windows 11） |
| 线程 | `RAYON_NUM_THREADS=4`（与官方 `latency_client.sh` / `latency_server.sh` 一致） |
| 拓扑 | 客户端与服务端同机回环 `127.0.0.1`，无跨机网络时延 |
| HE | SEAL 3.6.5，BFV，N=8192，`BFVDefault` 系数模数，128-bit |
| 定点 | `TenBitExpParams`：`MANTISSA_CAPACITY=3`（小数位）+ `EXPONENT_CAPACITY=8`（整数位），官方默认 |
| 模型 | 官方 MiniONN CIFAR-10 `7_approx`（5 层 3×3 卷积 + 2 层 1×1 降维 + 2 次平均池化 + FC，7 个激活层全部为多项式） |

## 3. CIFAR-10 MiniONN 主结果（每档 5 次独立运行，含重新 Keygen 与重新建连）

| 多项式层数 | client offline (s) | client online (s) | client 双向通信 offline | client 双向通信 online | client 峰值 RSS | server 峰值 RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 0（全 ReLU/GC） | 29.255 ± 1.288 | 3.329 ± 0.145 | 3240.0 MiB | 196.4 MiB | 2911 MiB | 5377 MiB |
| 1 | 30.901 ± 2.537 | 3.608 ± 0.630 | 3222.1 MiB | 195.3 MiB | 2900 MiB | 5354 MiB |
| 2 | 29.664 ± 0.984 | 3.503 ± 0.495 | 3147.5 MiB | 190.8 MiB | 2837 MiB | 5257 MiB |
| 3 | 23.760 ± 0.831 | 1.877 ± 0.051 | 1464.8 MiB | 88.8 MiB | 1557 MiB | 3135 MiB |
| 5 | 20.295 ± 0.372 | 0.948 ± 0.017 | 485.6 MiB | 29.3 MiB | 627 MiB | 1942 MiB |
| 6 | 19.634 ± 0.618 | 0.799 ± 0.022 | 189.8 MiB | 11.4 MiB | 383 MiB | 1517 MiB |
| **7（最佳配置）** | **19.318 ± 0.288** | **0.805 ± 0.029** | **116.4 MiB** | **6.93 MiB** | **347 MiB** | **1440 MiB** |

要点：

1. **多项式层越多，在线越快、通信越小**：ReLU 层走 GC，通信与内存随 ReLU 数量急剧膨胀；7 层全多项式是官方最优配置（在线 0.805 s，online 双向 6.93 MiB）。
2. 客户端 offline 的 19.3 s 中，客户端自身密钥生成仅 **376.5 ms**，其余为等待服务端线性层预处理（官方 trace 显示单个线性层 offline 达 9.8 s）。
3. 客户端 online 的 802.8 ms 中，7 个 Approx 层各约 **90–115 ms**（合计约 700 ms），7 个线性层合计约 5 ms。
4. 峰值内存：客户端 347 MiB / 服务端 1440 MiB（7 层配置）。

原始逐次数据：`delphi_minionn_raw.csv`；均值/标准差：`delphi_minionn_summary.csv`；代表性日志：`logs/`。

### 3.1 线程数敏感性（18 线程，3 次）

| 多项式层数 | offline (s) | online (s) | offline 通信 | online 通信 |
|---:|---:|---:|---:|---:|
| 0（全 ReLU） | 25.907 ± 0.632 | 2.103 ± 0.115 | 3239.5 MiB | 196.4 MiB |
| 7（全多项式） | 19.400 ± 0.158 | 0.773 ± 0.027 | 116.4 MiB | 6.9 MiB |

结论：多项式层路径几乎不受线程数影响（19.32 s@4 线程 vs 19.40 s@18 线程），说明其瓶颈在密文 I/O 与串行 SEAL 运算；ReLU/GC 路径对核数更敏感（offline −11.4%，online −36.8%）。**论文主表采用官方脚本的 4 线程口径**，18 线程数据仅作敏感性参考。原始数据：`delphi_minionn_18thread.csv`。

## 4. 官方 MNIST 示例（3 层配置 3 次，0 层配置 1 次）

| 多项式层数 | offline (s) | online (s) | offline 通信 | online 通信 |
|---:|---:|---:|---:|---:|
| 3（全多项式） | 2.208 ± 0.056 | 0.174 ± 0.010 | 52.1 MiB | 385 KiB |
| 0（全 ReLU） | 2.801 | 0.280 | 237.9 MiB | 11.7 MiB |

原始数据：`delphi_mnist_raw.csv`；日志：`logs/client_L*.log`。

注：官方 MNIST 示例（`construct_mnist`）使用**随机初始化权重**，仅用于验证 Pipeline 与测量时延/通信，不代表任何训练精度。

## 5. 精度与正确性

| 项 | 结果 | 来源 |
|---|---|---|
| 官方模型精度（Keras，10k 测试集） | 82.95% | `python/minionn/pretrained/7_approx/model_state` |
| 本机明文复算（100 张，seed 20261029） | 82.0%（82/100） | 本复现 |
| 本机密文 spot check（5 张） | 5/5 与标签及明文预测一致 | `logs/accuracy_spot_summary.txt` |

说明：官方 `minionn-accuracy` 批量校验工具存在一个与复现环境无关的端口复用缺陷——它在同一端口上反复派生 server/client，而未设置 `SO_REUSEADDR`，第二批图片起会因 TIME_WAIT 绑定失败并 panic（`ConnectionRefused`，`experiments/src/lib.rs:54`）。因此**未**用该工具跑全量密文精度；改用官方 `minionn-inference` 逐图做 5 张 spot check，并配合明文 100 张复算。此缺陷属官方测试脚手架问题，未修改其代码。

## 6. 与 LitePI 对比时的口径提醒（写入论文须注意）

1. **HE 方案不同**：Delphi 用 BFV + 定点（3 位小数 / 8 位整数）；LitePI 用 CKKS + 实数编码。两者不是逐位等价的安全/精度口径，只能做系统级对比。
2. **模型不同**：Delphi 基线是 MiniONN（3 卷积 + FC，CIFAR-10），LitePI 的 CIFAR 实验是更深的 64 通道 VGG 型网络；因此**单次推理总时延/总通信**只能在「各自最优模型」的意义上比较，不能按层直接换算。
3. **网络不同**：本基线是同机回环，不含 WAN 往返；LitePI 的 RTT 敏感性由 E3 单独建模。若论文要并列，需要标注 Delphi 为「本地回环、无网络时延」。
4. **offline 含 Keygen**：Delphi 的 19.3 s offline 包含一次完整 Keygen（376 ms）与全部线性层预处理；密钥可跨推理复用，因此该成本应视为一次性摊销项。
5. **客户端设备不同**：Delphi 基线在本机桌面 CPU 测；LitePI 的客户端开销（E2）在树莓派侧测。两者客户端设备不同，不能直接并列成「同设备客户端耗时」。
6. **线程口径**：Delphi 按官方脚本固定 4 线程（客户端/服务端同机共享核）；LitePI 服务端使用多核。论文须写明线程设置，否则时延不可比。

## 7. 复现命令（可重跑）

```bash
# 环境
export LD_LIBRARY_PATH=/root/libtorch/lib
export RAYON_NUM_THREADS=4

# 单次 CIFAR-10 MiniONN 7 层推理（当前目录需含 image.npy / class.npy）
cd /root/delphi/run
/root/delphi/rust/target/release/minionn-inference \
  --weights /root/delphi/python/minionn/pretrained/7_approx/model.npy --layers 7

# 官方双阶段延迟/通信基准（server 先起，client 后连）
/root/delphi/rust/target/release/minionn-server -l 7 -p 8000 &
/root/delphi/rust/target/release/minionn-client -i 127.0.0.1 -l 7 -p 8000

# 本次批量脚本（layers × 5 次 + 解析）
bash scripts/delphi_grid.sh
python3 scripts/delphi_parse.py
```
