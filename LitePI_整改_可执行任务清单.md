# LitePI 论文项目 · 可执行任务清单

> 依据：《LitePI论文项目审阅与完善清单》（2026-11-09，只读审阅）
> 落地为：带优先级、证据、执行动作与验收标准（✅）的可逐项勾选、可追踪的任务列表。
> 路径均相对仓库根目录 `/workspace`（原 `D:/王梓杨/项目组/wzy`）。
> 状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成。
> 说明：清单引用的行号以当前 `master` HEAD 为准（部分与审阅时 `sci-backend 6c9be8f` 有轻微偏移，已在括号内标注审阅原行号）。

---

## 0. 总览与推进顺序

| 阶段 | 主题 | 任务数 | 说明 |
|---|---|---:|---|
| P0 | 先修证据与正确性 | 6 | 阻断性，先于一切大实验 |
| P1 | 实验闭环 | 7 | 形成论文主要结论所需实验 |
| P2 | 外部效度与论文 | 3 | 收尾与文稿修订 |
| 附录 | Word 表格修订 + 数据快照 | — | 随 P0–P2 同步完成 |

推进规则：**P0 全部通过验收后，才进入 P1；P1 完成后进入 P2。** 不允许在 P0 未闭环时按"贡献已验证"口径投稿。

---

## 阶段 P0 —— 先修证据与正确性

### P0-1 冻结证据版本（可追溯基线）

- **问题/证据**：GitHub `sci-backend` HEAD `6c9be8f…`（2026-11-07）与本地 `wzy` HEAD 相同；`master` 为 `9487bea…`（2026-10-24），与本轮审计分支不同。引用仓库必须注明分支 + commit。
- **执行动作**：
  1. 选定冻结基线：`sci-backend/6c9be8f` 或修复后新 commit。
  2. 记录并入库：diff hash、扩展 hash、SEAL/SCI 版本、Python ABI、设备、线程数、wire/mask/F 参数。
  3. 更新 `MANIFEST`；旧记录只标 `stale`，**不覆盖、不删除**。
- **验收（✅）**：论文中每个表格数值都能回溯到唯一配置（commit + 参数 + 产物路径）。

### P0-2 修复 argmax 双份额常数偏移

- **问题/证据**：[secure_argmax.py](file:///workspace/analysis/secure_argmax.py#L54-L60)（审阅标注 `secure_argmax.py:54–60`）中，比较值用 `[(champion_score - q[i] - 1) % mask]`，两方各自对差值份额减 1，重构时**共减 2**。
- **反例**：server 份额 `[0,0]`、client 份额 `[1/256,0]`、`f=8` 时，比较值重构为 `−1`，而真实差为 `+1 LSB`；应选类别 0 却选 1。
- **执行动作**：
  1. 把公共常数 `−1` 只分配给**指定一方**（如 `−1 if ctx.party == 0 else 0`），使重构差值 = `true_diff − 1`。
  2. 统一 tie 规则（严格大于 vs 大于等于），并写入 `secure_argmax.py` 与协议设计文档。
- **验收（✅）**：唯一最大值相差 1 LSB 的反例选对；新增回归测试覆盖 0/1/2 LSB、负值、范围边界。

### P0-3 修复 planner 预算检查顺序

- **问题/证据**：[planner.py](file:///workspace/analysis/planner.py#L490-L523) 中 `feasible_schedules()` 只检查 `depth_ok / memory_ok / acc`，`constrained()` 选完才在 `plan_result()`（[L256-L264](file:///workspace/analysis/planner.py#L256-L264)、[L319](file:///workspace/analysis/planner.py#L319)）检查预算。（审阅标注 `planner.py:319–349、461–473`）
- **反例**：100ms 但超 MPC 预算的候选 vs 110ms 且满足预算的候选，原搜索选前者、最终 `feasible=false`，而后者明明可行。
- **执行动作**：
  1. 把 `max_mpc_elements / max_rtts / max_time_ms / max_memory_mb` 预算纳入候选过滤，而非事后检查。
  2. 与小规模穷举 oracle 对照验证一致性。
- **验收（✅）**：存在可行候选时**不**错误返回 `infeasible`；"约束最优"语义成立。

### P0-4 修复评测 schema（逐样本可追溯）

- **问题/证据**：[eval_hybrid_accuracy.py](file:///workspace/analysis/eval_hybrid_accuracy.py#L332) 将 `prediction_mismatches` 定义为 `pred != label`（审阅标注 `eval_hybrid_accuracy.py:359`），把"分类错误数"误作"密文与明文不一致数"，表 7-5 的 99.69%–100% 一致率来源链不成立。
- **执行动作**：
  1. 逐样本保存：`plain_pred`、`cipher_pred`、`label`、`margin`、误差、`sample_id`、`checkpoint_hash`。
  2. 分开报告：分类错误（Acc）、agreement、logit 误差、低 margin 翻转。
  3. 修正周报 8/64 口径：`00/01/10` 仅 **8 样本**，`11` 为 **64 样本**（单 seed 73）。
- **验收（✅）**：汇总值能从逐样本重算；分类错误与 HE 不一致**独立**呈现，不互相改写。

### P0-5 补公开复现链

- **问题/证据**：当前 [.gitignore](file:///workspace/.gitignore#L18) 已只忽略 `sci/build/`，且 [sci/litepi_sci.cpp](file:///workspace/sci/litepi_sci.cpp) 与 [sci/CMakeLists.txt](file:///workspace/sci/CMakeLists.txt) 已被 git 追踪（审阅时"忽略整个 sci"已部分解决）。但 [build_sci.sh](file:///workspace/scripts/build_sci.sh#L15-L16) 用 `git clone --depth 1`（**未锁 revision**），且路径写死为 `third_party/EzPC/SCI`。
- **执行动作**：
  1. 提交自研 SCI 源码/CMake/pybind 包装；仅忽略第三方源码与编译产物。
  2. 锁第三方 revision（EzPC/SCI 明确 commit + 校验和）。
  3. 权重不入 Git，但提供版本化下载/生成步骤 + hash。
  4. `build_sci.sh` 改为可配置路径、从明确版本获取、去掉对本地已存在 EzPC 的隐式依赖。
- **验收（✅）**：干净 `git clone` 后能构建并跑通一个最小真实 HE+SCI 测试。

### P0-6 决定安全与模型范围（统一口径）

- **问题/证据**：全文在"输入隐私 vs 双方隐私""模型库 vs 统一权重"两处口径未定，导致贡献/威胁模型互相矛盾。
- **执行动作**：
  1. 明确选择：输入隐私 或 双方隐私；模型配置库（路线 A）或 统一权重可切换模型（路线 B）。
  2. 建议：先收敛到**路线 A（模型配置库）**——更贴近现有实现与数据。
- **验收（✅）**：`Abstract / Threat Model / Design / Planner / Evaluation` 使用同一功能定义，无相互矛盾承诺。

---

## 阶段 P1 —— 实验闭环

### P1-1 E1 重测最小矩阵

- **执行动作**：三任务 × 四调度 × 五训练 seed，固定分层随机测试 ID；新 maskv3 + 真实 SEAL/SCI + 同构建；资源受限时统一 200–1000 配对样本并给置信区间，**不伪装全量**。
- **验收（✅）**：maskv3 各调度同样本多种子；同权重或模型库定义明确；配对的 agreement/Acc/误差分开报告。

### P1-2 E2/E4 统一计时与字节口径

- **执行动作**：直接计时 encode/encrypt/decrypt/post 与 MPC CPU（`wait` 另记）；全 channel 字节含 HE/SCI/控制帧，setup 基础 OT 另列；压缩同时报字节与 CPU 时间。
- **验收（✅）**：server compute / client compute / wait / network 互斥计时；压缩倍数注明范围（HE wire，不含 SCI），不写成全协议倍数。

### P1-3 E5 实际适应性验证

- **执行动作**：以 validation 选 `Acc_min` 及成本，test 最终评估；扫精度下限/RTT/带宽/内存，寻找真实选择交叉点；用留出实测检查成本预测误差。
- **验收（✅）**：找不到客户端目标收益就**降低该贡献**，不人为设常数造优势。

### P1-4 公平 Delphi 主基线

- **执行动作**：同输入任务、同容量/精度预算、同 client/server 设备与拓扑；允许 HE 方案不同但安全目标相同；分别报告一次性 setup、逐查询预处理、在线 CPU、在线等待。
- **验收（✅）**：不把"纯计算 vs 墙钟离线+在线""60% vs 83%"这类不对等口径写成加速倍数。

### P1-5 E3/E9 分组实测

- **执行动作**：完整配置扫描不跨机器挑赢家；记录实际 RTT/bandwidth/loss/jitter/`delay_mult`；删除旧"一消息一包"推理，保留 TCP 重传与 RTO 诊断。
- **验收（✅）**：按 device/topology/delay_mult 分组，禁止跨组选择"最优"。

### P1-6 安全桥接与 E10

- **执行动作**：验证新鲜/复用、范围边界、70-bit 掩码解码、份额 wrap、重启/重放/Q 预算；区分回放与真实协议攻击；补形式化模拟和有限精度界；AUC 仅作辅助。
- **验收（✅）**：一次性随机材料索引不复用；失败后不可继续消费旧材料；不把 AUC≈0.5 当作"所有成员推理不可行"。

### P1-7 ARM 混合部署

- **问题/证据**：SCI 现有 aarch64 编译受 x86 AES-NI 实现限制，Pi5 仅 HE 不等于完整 Hybrid 已部署。
- **执行动作**：补可移植后端，真实跑 `01/10/11`；失败则诚实缩小"真实边缘 Hybrid"结论。
- **验收（✅）**：要么给出真实 ARM Hybrid 结果，要么明确收回该结论。

---

## 阶段 P2 —— 外部效度与论文

### P2-1 E6 能耗

- **执行动作**：准备实测功率轨迹、采样频率/校准、空闲扣除、通信与计算同步标记、重复及积分脚本。
- **验收（✅）**：条件不具备时去除节能定量贡献，保留待测；不得用 CPU 估算/瞬时功率填实测。

### P2-2 设备/任务扩展

- **执行动作**：按实际可用硬件扩展设备档位；ECG 可选，若作核心医疗场景需补 MIT-BIH/PTB-XL 实验，否则把主叙事改为 UCI HAR 活动识别、ECG 只作潜在应用。
- **验收（✅）**：不把原方案"4 档设备"强写为已完成。

### P2-3 正式文稿修订

- **执行动作**：
  - 重写 Abstract 与贡献（见附录 C 创新主张收敛）；
  - Related Work 按 DELPHI 原文；
  - Design 写 SCI 真实原语与模型库；Planner 删旧组合罚项/外推 VGG；
  - Security 补限定定理；Evaluation 换当前证据；
  - Discussion 更新 v3/conv3 已实现但证据有限；
  - 整理真实 BibTeX 与引用，删除占位符、内部修订话语、无法支撑的倍数。
- **验收（✅）**：Abstract/Threat Model/Design/Planner/Evaluation 口径一致，无"首次单服务器""首次混合激活 planner""解密即份额新原语""dummy 隐藏服务器自己调度"等强结论。

---

## 附录 A —— Word 表格修订清单

| 表格 | 发现 | 动作 |
|---|---|---|
| 表7-4 | canonical 五种子数值基本一致 | 保留数值，修订"同模型/多项式损失"解释；补验证集与训练公平性 |
| 表7-5 | 一致率证据与指标定义缺口 | 补 `plain_pred`；分开 old/maskv3、样本数、seed、backend、配置 |
| 表7-6 | 深网多配方/多 epoch/种子口径混用 | 按每行实际 JSON 核对 epoch/lr/seed；未齐五种子则明确三种子 |
| 表7-7 | 3.6–3.8ms 属旧栈；11-06 当前栈约 8.5–8.7ms | 全表按一个构建重制，Pi5 另列；不得只改均值保留旧 std |
| 表7-8 | 边际成本为 E1 拟合；"OT 未记录"已落后 | 改清计时边界；更新 SCI 在线/基础 OT 分项 |
| 表7-9 | 流式内存仍写 112MB，最新独立进程约 121MB | 引用新 client_only 证据；分离 setup 峰值与 online 峰值 |
| 表7-10/11 | baseline 序列化仅 HE wire，不含 SCI | 保留范围声明；不把 compact HE 压缩倍数写成全协议倍数 |
| 表7-12 | 已区分主通道 1 与端到端约 6 | 保留区分；不称 6.025 为严格整数轮数，补抓包/因果层测量 |
| 表7-15 | 丢包近似"一消息一包"，0.25s 无 TCP 依据 | 改用最新 netem 分设备实测；旧模型仅作已弃用方法说明 |
| 表7-16 | 00 字节 1.70MB 为旧 M3 选错；当前约 1.18MB | 从按 schedule 的 wire 表重生成，重新校准时延 |
| 表7-17、贡献(2) | 5 层 VGG 组合罚项与外推已移除 | 替换为真实 conv3 模型 B；只测 4/16 调度则不称全空间最优 |
| 表7-20、Discussion | 仍述 double 精度限制及 v3 未完成 | 更新 v3 条件性统计界/解码误差；不把有限样本界宣称普适安全 |
| 表7-21 | 结果待入库，但 E10 已有五随机攻击 seed 产物 | 按真实攻击口径重制；不称五 target 训练种子或 shadow-model 证据 |
| 表7-22 | 截断算子微基准，非当前 SCI ReLU 必要截断 | 移附录/扩展，或加真实协议消融 |
| 表7-24 | 01/10 都称 Adaptive，但默认 planner 可能选 00 | 按实际 profile/Acc_min 标出选择结果；固定配置与 Adaptive 分开 |
| 表7-25 | Delphi 墙钟离线+在线 对 LitePI 纯计算；60% vs 83% | 建互斥同口径指标；不给公平加速倍数 |
| 表7-26 | Delphi 原语/全多项式回退/随机材料描述有误 | 依 DELPHI 原文重写，不将 GC tables 下载划入 online |
| 表7-27、Discussion | 深网/v3 仍列未实现 | 更新为已实现待验证，保留 E6/ARM 混合部署/公平基线等真实缺口 |

---

## 附录 B —— 关键数据快照

### B.1 canonical 明文结果（五 seed 均值 ± std，单位 %）

| 数据集 | 00: P₂/P₂ | 01: P₂/ReLU | 10: ReLU/P₂ | 11: ReLU/ReLU |
|---|---:|---:|---:|---:|
| MNIST | 97.972±0.293 | 97.904±0.110 | 97.886±0.201 | 97.708±0.091 |
| CIFAR-10 灰度 | 60.592±0.492 | 59.360±0.768 | 62.796±0.766 | 58.604±0.464 |
| UCI HAR | 90.336±0.930 | 88.666±1.118 | 90.811±0.511 | 89.752±0.990 |

> 注：每种调度**重新训练**，因此只支持"模型族比较"，不能证明"同权重近似损失"。

### B.2 maskv3 样本数（正确数/样本，单 seed 73）

| 数据集 | 00 | 01 | 10 | 11 |
|---|---:|---:|---:|---:|
| MNIST | 8/8 | 8/8 | 8/8 | 61/64 |
| CIFAR-10 | 6/8 | 6/8 | 6/8 | 33/64 |
| UCI HAR | 8/8 | 8/8 | 8/8 | 51/64 |

> 11 月 6 日周报把所有调度统一写为 64 样本，**必须更正**。

### B.3 E2 baseline 序列化（当前栈）

- 上行 917,625 B、下行 262,265 B，合计 **1,179,890 B**。
- 密钥项 184,634,920 B（不含 sk）；公开 context 带头部 184,634,972 B。
- root 协议约 464KB 的"理想尺寸"不对应 SEAL RNS 序列化；未压缩 RNS 主体 = `2×N×剩余素数个数×8B`。

---

## 附录 C —— 创新主张收敛

**建议保留为候选贡献**（每项须独立实测 + 对照）：
1. 客户端公钥/评估密钥流式生成与上传；
2. CKKS 线格式压缩；
3. 资源目标下的模型配置库选择；
4. 真实多查询随机材料管理及有条件统计分析；
5. 更深模型中的协议/参数可行性权衡。

**不应直接保留的强结论**：
- 首次单服务器；首次混合激活 planner；解密即份额天然新原语；
- 所有链路上 Adaptive Pareto 支配；每 hybrid 层完整 1 RTT；
- 比 Delphi 客户端快若干数量级；dummy 隐藏服务器自己的调度。

---

## 附录 D —— 红线（禁止做法）

1. 不把 Python P-256 结果写成 SCI/CrypTFlow2 最终结果。
2. 不把 fast RGB 模型写成 canonical CIFAR 主模型。
3. 不用固定 mask 域冒充 M2。
4. 不在输出档位 B 中把 logits 解密给客户端。
5. 不在 `check_protocol_compliance.py` 未通过时声称严格协议已完成。
6. 不提交 `datasets/`、`datasets_organized/`、`datas/` 中的数据集或权重。
7. 不覆盖/删除旧实验记录（红线①），只标 stale。
