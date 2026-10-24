# LitePI Codex → Trae 交接

交接日期：2026-10-24
仓库：`D:\王梓杨\项目组\wzy`
远程：`https://github.com/MrWangneverplay/LitePI.git`
分支：`master`

这份文件汇总此前 Codex 聊天中的关键决定、代码状态、验证命令和未完成项。
它不包含 WiFi 密码、服务器密码、token 或其他凭证。

## 1. 当前目标

LitePI 的最终实现必须同时对照以下两份权威文档：

- `D:\王梓杨\项目组\实验方案设计.md`
- `D:\王梓杨\项目组\LitePI 协议设计.md`

HE-only、Hybrid-all、Adaptive 是同一个协议的三个配置点，不是三个协议。

必须遵守的总口径：

- `s=0`：服务端执行 HE-P2，客户端无交互。
- `s=1`：执行 Hybrid-ReLU，每个 hybrid 层增加一个逻辑往返。
- `s=00`：在线 2 轮；`K_h` 个 hybrid 层时为 `2 + K_h` 轮。
- 消息链：Setup/M0 → M1 → 每个 hybrid 层的 ①/② → M3。
- M1：每个 `(query, layer)` 使用独立 PRG 掩码。
- M2：掩码域必须覆盖该层训练激活分布。
- M3：查询计数失步必须中止并重新 Setup。
- 输出档位 A：客户端获得 logits。
- 输出档位 B：客户端只获得类别索引。
- 最终软件栈：Python 3.11、TenSEAL/SEAL、pybind11 C++、
  SCI/CrypTFlow2 silent OT/VOLE、128-bit CKKS 参数校核。

## 2. 已完成

### Canonical 三模型

服务端已经从随机小 MLP 切换到 `analysis/train_hybrid.py` 的三个 canonical
架构：

| 数据集 | 非线性维度 | 编码槽位 | 参数量 |
|---|---:|---:|---:|
| MNIST | 256 → 64 | 4096 | 17,298 |
| CIFAR-10 灰度 | 1200 → 128 | 3200 | 155,330 |
| UCI HAR | 756 → 128 | 2016 | 98,006 |

入口：

- `analysis/canonical_net.py`
- `analysis/check_canonical_contract.py`

fast RGB 4-conv 网络只能作为扩展实验，不能替代 canonical CIFAR 主模型。

### 协议模式与调度

`analysis/pi_server.py` 和 `analysis/pi_client.py` 已支持：

- `s=00/01/10/11`
- HE-P2 与 Hybrid-ReLU 混合调度
- planner JSON 通过 `--plan`
- `--obfuscate-schedule` 的 §7.2 dummy ciphertext pair

验证：

```powershell
python analysis/check_protocol_modes.py
```

### M1–M3 与 M2 域

- `analysis/mask_session.py`
- `analysis/export_mask_domains.py`
- `analysis/check_mask_session.py`

服务器会校验 mask-domain JSON 的 dataset、schedule、seed 和层维度，防止加载
错档实验的统计。

生成 M2 文件：

```powershell
python analysis/export_mask_domains.py --dataset cifar10 --schedule 10
```

### 输出档位 B

- `analysis/secure_argmax.py`
- `analysis/check_output_modes.py`

实现为顺序锦标赛式安全 argmax：`C-1` 次安全比较，复杂度 `O(C·b)`。
客户端最终只组合索引份额，不接收完整 logits。

### Planner 与成本模型

- `analysis/planner.py`
- `analysis/client_cost.py`
- `analysis/check_client_cost.py`

Planner 输出 `s / p / packing / output_mode`，canonical 精度使用五种子 E1
实测值，不再使用 fast-net 的通用惩罚公式代替 canonical 模型。

## 3. 关键实验数字

canonical CIFAR-10 五种子 E1 均值：

| 调度 | 精度 |
|---|---:|
| `00` | 60.59% |
| `01` | 59.36% |
| `10` | 62.80% |
| `11` | 58.60% |

fast RGB 4-conv 扩展结果约为 84% 以上，但不属于 canonical 主实验口径。

Pi Python 参考实现的早期结果包括：

- 64→32→10 小模型冷启动约 10.45 s。
- warm-query 约 1.74–1.97 s。
- 32 元素 MPC 约 1.34 s。
- 1328 元素批处理 ReLU 约 43.28 s、82 条消息。

这些只能标记为 `reference-only`，不能写成最终论文部署结果。

## 4. 已通过的检查

```powershell
python analysis/check_canonical_contract.py
python analysis/check_protocol_modes.py
python analysis/check_output_modes.py
python analysis/check_mask_session.py
python analysis/check_client_cost.py
python analysis/secure_argmax.py
```

结果均为 PASS。

严格门禁仍需运行：

```powershell
python analysis/check_protocol_compliance.py --json
```

在补齐下述缺口前，该命令必须保持 FAIL。

## 5. Trae 接手后的第一优先级

1. 补齐 SCI/CrypTFlow2 的 silent OT/VOLE 与 pybind11 C++ 后端。
2. 将 MPC 后端从 Python P-256 切换为 SCI/CrypTFlow2，并把每层物理消息数压到
   `r_h=1`。
3. 切换到 Python 3.11。
4. 建立正式 CKKS profile，使用 N、模数链和 SEAL/TenSEAL estimator 产物校核
   至少 128-bit 安全。
5. 重跑 canonical MNIST/CIFAR/HAR 的 M2、HE-only、Hybrid 和输出档位 A/B。
6. 用正式 C++/SCI 实测回填客户端计算、内存、能耗、通信字节和轮数。

## 6. 明确禁止的错误做法

- 不要把 Python P-256 结果写成 SCI/CrypTFlow2 最终结果。
- 不要把 fast RGB 模型写成 canonical CIFAR 主模型。
- 不要用固定 mask 域冒充 M2。
- 不要在输出档位 B 中把 logits 解密给客户端。
- 不要在 `check_protocol_compliance.py` 未通过时声称严格协议已完成。
- 不要提交 `datasets/`、`datasets_organized/`、`datas/` 中的数据集或权重。

## 7. Git 与数据说明

整理后的 `datasets_organized/` 约 705 MB，已加入 `.gitignore`，只保留在本机。
代码、结果 JSON、汇总文档和实验记录会进入 Git；大体积数据集和权重不进入 Git。
