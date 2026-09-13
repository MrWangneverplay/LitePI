# 同态加密友好神经网络 — 三个数据集(MNIST / CIFAR-10 / UCI HAR)设计参考

> 本文档汇总本仓库三个 **HE(同态加密)友好 CNN** 的网络结构、加密参数、几何核算与训练/推理结果,供后续开发与论文写作参考。
> 仓库根目录:`/workspace/HE_AI`,依赖 `tenseal 0.3.16`(CKKS)、PyTorch、torchvision。

---

## 1. 项目概况

| 项 | 值 |
| --- | --- |
| 方案 | CKKS(TenSEAL 0.3.16,CPU) |
| 数据流 | 客户端预处理并加密 → 服务端在密文上推理(权重保持明文)→ 客户端解密取 argmax |
| 目录 | `model/` 脚本、`datasets/` 数据、`datas/` 权重与日志、`TenSEAL/` 源码参考 |
| 主脚本 | `model/mnist_cnn_enc.py`、`model/cifar10_cnn_enc.py`、`model/uci_har_cnn_enc.py` |
| 明文对照 | `model/mnist_cnn.py`、`model/cifar10_cnn.py`(常规 CNN,见 §7) |

## 2. 统一设计范式(为什么“HE 友好”)

同态计算只支持**加法与乘法**,因此三个网络统一采用同一范式:

- **激活**:以平方激活 f(x)=x² 替代 ReLU / MaxPool(明文训练与密文推理逐位一致);
- **拓扑**:`Conv2d → x² → 全连接 → x² → 全连接`,共 **5 次乘法深度**;
- **明文侧预处理**(归一化/灰度化)保证密文输入与明文训练输入一致;
- **加密算子映射**:卷积 = `conv2d_im2col`(im2col 编码 + 对角化);全连接 = `mm`(Halevi–Shoup 对角线法);多通道拼接 = `pack_vectors`;激活 = `square_`。

### 2.1 统一 CKKS 加密参数

| 参数 | 取值 |
| --- | --- |
| poly_modulus_degree | 8192 |
| coeff_mod_bit_sizes | [40, 21, 21, 21, 21, 21, 21, 40] |
| 单密文槽位数 | 4096 |
| global_scale | 2^21(约 21 bit 精度) |
| 乘法深度(模型 / 预算) | 5 / 6 |
| 旋转密钥 | generate_galois_keys()(pack/mm/conv 需要) |

**核心约束**:输入加密向量长度必须 ≤ 4096 槽(否则被切成多密文,`conv2d_im2col`/`mm` 全部禁用)。

## 3. MNIST(`model/mnist_cnn_enc.py`)

- 输入:1×28×28 灰度,像素归一化 [0,1](ToTensor)。
- 数据集:torchvision MNIST,自动下载;seed=73。

### 3.1 网络结构

| 层 | 配置 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | — | 1×28×28 | 归一化 [0,1] |
| Conv2d | 1→4 通道,核 7×7,stride 3,无 padding | 4×8×8 | (28-7)/3+1=8 |
| Square | f(x)=x² | 4×8×8 | — |
| Flatten | — | 256 | 4×8×8 |
| FC1 | 256→64 | 64 | — |
| Square | f(x)=x² | 64 | — |
| FC2 | 64→10 | 10 | 类别输出 |

### 3.2 槽位几何与参数量

- im2col 窗口:8×8 = **64**;每窗 next_pow2(49)=64 槽 → 编码总长 64×64 = **4096**(恰为单密文上限)。
- pack 后 4×64 = 256 槽;FC 输出 64 → 10。
- 参数量:总 **17,298(0.017 M)**;Conv 200(4×(7×7+1))、FC1 16,448、FC2 650。
- 乘法深度:conv+square+fc+square+fc = **5**。

### 3.3 训练与结果

| 项 | 值 |
| --- | --- |
| 轮数 / 优化器 / lr / batch | 10 / Adam / 1e-3 / 64 |
| 明文测试精度(存档 log) | **97.64%**(Epoch 10/10) |
| 产物 | `datas/mnist_he_cnn_train.{pt,log}` |

## 4. CIFAR-10(`model/cifar10_cnn_enc.py`)

- 明文域预处理:RGB(3×32×32)→ **ITU-R BT.601 亮度灰度**(权重 0.2989/0.5870/0.1140,等价固定 1×1×3 卷积)→ z-score(mean≈0.4808,std≈0.2465)。
- 训练增强(仅明文):RandomCrop(padding=4)+ 水平翻转;seed=73。

### 4.1 网络结构

| 层 | 配置 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | 1×32×32 | — | 灰度(客户端明文转换) |
| Conv2d | 1→12 通道,核 5×5,stride 3,无 padding | 12×10×10 | (32-5)/3+1=10 |
| Square | f(x)=x² | 12×10×10 | — |
| Flatten | — | 1200 | 12×100 |
| FC1 | 1200→128 | 128 | — |
| Square | f(x)=x² | 128 | — |
| FC2 | 128→10 | 10 | 类别输出 |

### 4.2 槽位几何与参数量

- im2col 窗口:10×10 = **100**;每窗 next_pow2(25)=32 槽 → 编码总长 100×32 = **3200 ≤ 4096**(单密文)。
- pack 后 12×100 = **1200 ≤ 4096**。
- 参数量:总 **155,330(0.155 M)**;Conv 312、FC1 153,728、FC2 1,290。
- 乘法深度:**5**。

### 4.3 训练与结果

| 项 | 值 |
| --- | --- |
| 轮数 / 优化器 / lr / batch / wd | 40 / AdamW + CosineAnnealing / 1e-3 / 128 / 1e-4 |
| 明文(灰度)测试精度(存档 log) | **62.80%**(Epoch 40/40) |
| 产物 | `datas/cifar10_he_cnn_train.{pt,log}` |
| 备注 | 灰度 32×32 + 仅平方激活 + 全连接压缩,分辨率信息损失大;62.8% 属 HE 演示常见水平 |

## 5. UCI HAR(`model/uci_har_cnn_enc.py`)

- 输入:**原始 Inertial Signals 窗口** 1×128×9(128 时间步 × 9 信号通道:body_acc/body_gyro/total_acc 各 x/y/z),**非** 561 维预计算特征。
- 预处理:按 9 通道分别 z-score(统计量缓存 `datas/uci_har_he_stats.npz`,训练集均值/方差)。
- 数据规模:训练 7352 / 测试 2947 窗口,6 类活动(WALKING…LAYING);数据集位于 `datasets/UCI_HAR/UCI HAR Dataset/`(若缺失脚本会打印下载引导后退出)。

### 5.1 网络结构

| 层 | 配置 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | 1×128×9 | — | 时间 128 × 通道 9 |
| Conv2d | 1→12 通道,核 3(时间)×9(通道),stride (2,1),无 padding | 12×63×1 | 核宽覆盖全部通道,仅时间轴滑动 |
| Square | f(x)=x² | 12×63×1 | — |
| Flatten | — | 756 | 12×63 |
| FC1 | 756→128 | 128 | 间有 Dropout(0.3,仅明文训练) |
| Square | f(x)=x² | 128 | — |
| FC2 | 128→6 | 6 | 6 类活动 |

### 5.2 槽位几何与参数量

- im2col 窗口:`(128-3)/2+1 = 63`(通道方向收敛为 1 个窗口);每窗 next_pow2(27)=32 槽 → 编码总长 63×32 = **2016 ≤ 4096**(单密文)。
- pack 后 12×63 = **756 ≤ 4096**。
- 参数量:总 **98,006(0.098 M)**;Conv 336(12×(3×9+1))、FC1 96,896、FC2 774。
- 乘法深度:**5**。

### 5.3 训练、回退机制与结果

- 训练:30 epochs(默认)/ AdamW + CosineAnnealing / lr 1e-3 / batch 128 / wd 1e-4;seed=73。
- **自动回退**:若明文测试精度 ≤ 0.70,自动扩为 channels=16、hidden=192、epochs=max(60,·) 重训(当前存档为 12/128/30 即达标,未触发回退)。
- 结果(存档 log):明文测试精度 **89.51%**(PASS > 0.70);加密 vs 明文逐样本 **3/3 一致**,单样本密文推理 **约 7 s**(CPU)。
- 产物:`datas/uci_har_he_cnn_train.pt`(含 state_dict + channels/hidden)、`datas/uci_har_he_cnn_train.log`、`datas/uci_har_he_stats.npz`。

## 6. 汇总对比

| 数据集 | 输入形状 | 网络拓扑 | 乘法深度 | 参数量 | 编码槽位 | 明文测试精度 |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 1×28×28 | Conv(4,7×7,s3)→sq→FC(64)→sq→FC(10) | 5 | 17.3 K | 4096(恰满) | 97.64% |
| CIFAR-10 | 1×32×32(灰度) | Conv(12,5×5,s3)→sq→FC(128)→sq→FC(10) | 5 | 155.3 K | 3200 | 62.80% |
| UCI HAR | 1×128×9 | Conv(12,3×9,s2)→sq→FC(128)→sq→FC(6) | 5 | 98.0 K | 2016 | 89.51% |

## 7. 关键实现细节与坑(重要)

1. **TenSEAL 0.3.x `im2col_encoding` 参数交换怪癖**:python 包装层会把 (kernel_n_rows, kernel_n_cols) **交换后**传给 C++ 后端。正方形核(MNIST 7×7、CIFAR 5×5)无感知;UCI HAR 非对称核 3×9 必须补偿——`encrypted_predict` 中以 `(9, 3)` 调用(即 `KERNEL_SHAPE[1], KERNEL_SHAPE[0]`)。否则 C++ 按 9×3 切窗得到 240 窗口、编码 7680 槽 > 4096,`conv2d_im2col`/`mm` 被禁用并告警。
2. **单密文硬约束**:任何加密向量长度 > 4096 槽即切分并禁用矩阵类算子;设计网络时必须先核算 `im2col 窗口数 × next_pow2(kernel 面积) ≤ 4096`,且 `输出通道数 × 窗口数 ≤ 4096`。
3. **权重形状**:`EncConvNet` 中全连接权重取 `weight.T`(明文矩阵按行组织);卷积核按 `(out_channels, *kernel)` 传入 `conv2d_im2col`。
4. **Dropout 仅明文训练**:eval 模式下消失,不影响密文前向,明文与密文输出逐位可比。
5. **明文对照模型**:`model/mnist_cnn.py`、`model/cifar10_cnn.py` 为常规 ReLU 网络(训练产物 `datas/*_cnn_train.*`),用于在论文中对比「HE 约束带来的精度损失」。

## 8. 复现命令

```bash
cd /workspace/HE_AI
python model/mnist_cnn_enc.py                      # MNIST:默认 10 epochs + 3 个加密样本
python model/cifar10_cnn_enc.py --num-samples 5    # CIFAR-10:灰度路线
python model/uci_har_cnn_enc.py                    # UCI HAR:30 epochs + 3 个加密样本
python model/uci_har_cnn_enc.py --train --num-samples 5   # 强制重训并验证加密一致性
```

各脚本均含 CLI:`--train`(强制重训)、`--epochs`、`--batch-size`、`--lr`、`--num-samples`(加密样本数)。已存在存档权重时默认直接加载并进入加密演示。
