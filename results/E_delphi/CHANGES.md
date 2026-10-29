# Delphi 官方实现复现记录（E_delphi）

本文件逐条记录为在本地跑通 Delphi 官方实现所做的**全部**改动。凡未列于此的代码与参数均为官方仓库原样。

## 1. 代码来源与版本

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/mc2-project/delphi |
| commit | `92bc0071fa11570df6b048ae0f6937ced249bb5a` (2022-11-26) |
| 本地路径 | WSL2 `/root/delphi` |
| 构建产物 | `/root/delphi/rust/target/release/{minionn-server,minionn-client,minionn-inference,minionn-accuracy}` |

## 2. 工具链与环境

| 项 | 值 | 说明 |
|---|---|---|
| rustc | 1.101.0-nightly (41089d07a 2026-10-27) | 官方 README 要求 nightly；本复现使用当前最新 nightly，未锁老版本 |
| cargo | 1.101.0-nightly (cad07cf8e 2026-10-23) | |
| OS | WSL2 Ubuntu（Windows 11 主机） | |
| CPU | Intel Core Ultra 5 125H，18 逻辑核 | 客户端/服务端同机回环，非两机 |
| 内存 | 15 GB | |
| libclang | `/usr/lib/llvm-14/lib`（`LIBCLANG_PATH`） | bindgen 依赖 |
| libtorch | `/root/libtorch/lib`（运行时 `LD_LIBRARY_PATH`） | tch 0.1.3 下载，链接成功；运行时必须显式给出，否则 `libtorch_cpu.so` 加载失败 |
| 构建变量 | `CARGO_NET_GIT_FETCH_WITH_CLI=true` | git 依赖走 CLI |

## 3. 源码改动（逐条）

> 全部改动仅为「让 2022 年的代码在 2026 年的工具链/离线环境下可编译可运行」，**未修改任何密码学逻辑、网络结构、实验参数或计时点**。

### 3.1 `rust/protocols-sys/build.rs`（唯一功能性改动，原始文件备份为 `build.rs.orig`）

官方 `build.rs` 通过 CMake `FetchContent` 在构建期从 GitHub 拉取 SEAL / ms-gsl / zstd。本复现的构建环境无法在构建期稳定访问这些仓库，因此：

1. 用 `FETCHCONTENT_SOURCE_DIR_SEAL` / `FETCHCONTENT_SOURCE_DIR_MSGSL` / `FETCHCONTENT_SOURCE_DIR_ZSTD` 指向本地已存在的源码目录：
   - SEAL 3.6.5：`c++/build/_deps/seal-src`
   - ms-gsl 3.1.0：`c++/build/_deps/seal-src/thirdparty/msgsl-src`
   - zstd 1.4.5（commit b706286）：`c++/build/_deps/seal-src/thirdparty/zstd-1.4.5`
2. 增加 `SEAL_USE_ZSTD=ON`。原因：`c++/src/lib/interface.cpp` 使用 SEAL 默认 `save/load` 序列化密文，该默认实现依赖 zstd 压缩；若关闭 zstd，C++ 接口与 Rust 端的密文字节格式不一致。

其余（链接 `seal-3.6` / `DelphiOffline`、bindgen 生成 bindings）保持官方逻辑不变。

### 3.2 lint 兼容改动（无语义影响）

2026 年的编译器把两处官方代码视为硬错误（`#![deny(...)]`）：

| 文件 | 改动 |
|---|---|
| `rust/algebra/src/fields/models/fp_32.rs` | 删除 `impl Neg` 里 `fn neg` 上的 `#[must_use]`（1 行） |
| `rust/algebra/src/fields/models/fp_32_lazy.rs` | 同上（1 行） |
| `rust/algebra/src/fields/models/fp_64.rs` | 同上（1 行） |
| `rust/algebra/src/fields/models/fp_256.rs` | 同上（1 行） |
| `rust/algebra/src/fields/models/fp_384.rs` | 同上（1 行） |
| `rust/algebra/src/lib.rs` | `private_in_public` → `private_interfaces`（lint 更名） |
| `rust/crypto-primitives/src/lib.rs` | 同上 |

说明：删除 `fn` 上的 `#[must_use]` 不改变行为（返回值仍是 `Self`），仅规避新版 `unused_attributes` 硬错误；`struct` 与 `trait` 声明上的 `#[must_use]` 全部保留。`rust/Cargo.lock` 为首次构建自动生成（官方未提交锁文件）。

## 4. 运行参数（与官方脚本一致）

- 线程：`RAYON_NUM_THREADS=4`（官方 `latency_client.sh` / `latency_server.sh` 的设定）
- 定点参数：`EXPONENT_CAPACITY=8`、`MANTISSA_CAPACITY=3`（`experiments/src/lib.rs` 默认的 `TenBitExpParams`，未改）
- HE 后端：SEAL 3.6.5 BFV，`POLY_MOD_DEGREE=8192`，`CoeffModulus::BFVDefault(8192)`（`protocols-sys/c++/src/lib/interface.h` 默认，未改）
- 模型：官方预训练 MiniONN，`python/minionn/pretrained/7_approx/model.npy`（7 个多项式激活层）
- 客户端/服务端：同机回环（127.0.0.1），非跨机

## 5. 复现脚本（本次新增，不属于官方仓库）

| 文件 | 作用 |
|---|---|
| `scripts/delphi_bench.sh` | 单次 server+client 配对运行，`/usr/bin/time -v` 记录 wall/RSS |
| `scripts/delphi_grid.sh` | layers ∈ {0,1,2,3,5,6,7} × 5 次 |
| `scripts/delphi_parse.py` | 从官方 trace 日志抽取 offline/online 时间与双向字节数，输出 CSV |
| `scripts/mnist_bench.sh` | 官方 MNIST 示例（layers=3 ×3 次，layers=0 ×1 次） |
| `scripts/mnist_parse.py` | 解析 MNIST 日志为 CSV |
| `scripts/delphi_grid18.sh` | 18 线程敏感性对照（layers ∈ {0,7} × 3 次） |
| `scripts/parse18.py` | 解析 18 线程对照数据 |

另外两个一次性辅助脚本（`scripts/delphi_gen_image.py` 生成单图 quick-start 的 `image.npy`/`class.npy`；`scripts/delphi_gen_val.py` 生成 100 张 CIFAR 测试图、`classes.npy` 与明文预测 `plaintext.npy`）均等价于官方 `generate_test_images.py` 的逻辑（后者依赖 TensorFlow），改为用 pickle 直接读 CIFAR-10 官方 `test_batch`，并用官方 `model.npy` 权重做明文复算。明文复算精度 82.0%（82/100，seed 20261029），与官方 `model_state` 记录的 82.95% 一致。

## 6. 已知的官方脚手架缺陷（未修改其代码）

`minionn-accuracy`（`rust/experiments/src/validation/validate.rs`）在同一端口（`base_port + port_off`，`base_port=8001`）上按批反复派生 server/client，但 `server_connect` 未设置 `SO_REUSEADDR`。第一批之后端口处于 TIME_WAIT，绑定失败，客户端随即在 `experiments/src/lib.rs:54` 以 `ConnectionRefused` panic。因此全量密文精度未用该工具跑；改用官方 `minionn-inference` 做 5 张 spot check（5/5 与明文预测和标签一致）。此为官方测试脚手架问题，与密码学实现无关，未修改其源码。

## 7. 数据口径

- `offline_s` / `online_s`：官方 trace 中 `End: Client offline phase` / `End: Client online phase`，**客户端视角**；offline 含客户端 Keygen。
- 通信量：客户端 `CountingIO` 统计的 Read/Wrote 字节（不含 TCP 头）。
- `client_rss_kb` / `server_rss_kb`：`/usr/bin/time -v` 的 `Maximum resident set size`。
- 全部数据为 5 次独立运行（每次重新建连、重新 Keygen），未做预热丢弃。
