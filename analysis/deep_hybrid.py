"""Deep-network ciphertext path: hybrid layers with client-side im2col re-encoding.

The canonical networks (K=2) encode im2col once, for the first convolution.
A deep network (fast / VGG-lite, 4 convs) needs every later convolution to
see its input in im2col form too.  After a hybrid ReLU the activation exists
only as additive shares (c_c at the client, c_s at the server), and im2col is
a linear re-indexing, so each party can lay out its own share:

    client:  rows_c = im2col(c_c)   -> encrypt block by block (②_k, now B cts)
    server:  rows_s = im2col(c_s)   -> plaintext; ct_b + rows_s,b = Enc(im2col(a))
    server:  Z = W_flat @ rows      via litepi_seal.matmul_rows (one level)

This keeps protocol §4.1 intact (the client re-encrypts its share, the server
adds its own) - only the share's layout changes, and step 5's "ciphertext +
server share" becomes "per-block ciphertext + per-block server share".  The
price is upload: B = ceil(C_in * k^2 / R) ciphertexts per layer instead of one
(R = slots / windows rows per ciphertext).

本模块的旧协议函数仅处理 hybrid 后的卷积。DeepNet 已通过融合的
linear_sparse 支持 P2 -> (pool +) conv，不能用新增校验默认拒绝该旧路径。
下方 B1 接口独立提供五位 schedule 校验与密文 im2col；模型身份未确认，
未接入 runner、模型加载或 DeepNet。显式拆出 im2col 会额外消耗一层，
不能套用旧融合路径的深度预算，也未验证其 P2 系数折叠和后续卷积。

Layout: a ciphertext holds R rows of W = windows slots; row t is im2col tap
t = (c_in, ki, kj) over all output positions; the output has one row per
output channel, i.e. PyTorch's (C, H, W) flatten order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class P2BeforeConvUnsupported(ValueError):
    """独立 B1 校验未获准使用 P2 -> conv；不限制旧 DeepNet 路径。"""


class DepthBudgetExceeded(ValueError):
    """两次 hybrid 刷新之间所需的 rescale 深度超过预算。"""


SCHEDULE_LAYERS = ("conv1", "conv2", "conv3", "conv4", "fc1")


def parse_schedule(schedule) -> tuple[int, ...]:
    """严格五位，依次对应 conv1..conv4、fc1 后的激活；不做宽松转换。"""
    if isinstance(schedule, str):
        valid = len(schedule) == 5 and all(b in "01" for b in schedule)
    elif isinstance(schedule, (tuple, list)):
        valid = len(schedule) == 5 and all(
            isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_))
            and b in (0, 1) for b in schedule)
    else:
        valid = False
    if not valid:
        raise ValueError("schedule 必须是严格五位 0/1（conv1..conv4、fc1）")
    return tuple(int(b) for b in schedule)


@dataclass(frozen=True)
class ScheduleOp:
    """由调用方显式命名的实际 op 链生成，不代表已确认的模型架构。"""

    name: str
    bit: int
    linear_index: int
    activation_index: int
    next_linear_index: int
    next_linear_name: str
    next_linear_op: str
    pool_indices: tuple[int, ...]
    linear_depth: int = 1
    p2_depth: int = 1


def schedule_ops(schedule, ops: list[dict]) -> tuple[ScheduleOp, ...]:
    """读取显式 op 元数据，不根据五位 schedule 猜 pool、卷积或输出头。

    ops 使用 DeepNet 的 conv/act/pool/fc 格式；线性 op 还须有唯一 name。
    五个激活前的线性 name 必须是 conv1..conv4、fc1，输出头也须显式命名。
    不需要权重，也不加载模型；缺失或不同结构直接报错，不补造 op。
    """
    bits = parse_schedule(schedule)
    if not ops or ops[0].get("op") != "conv":
        raise ValueError("必须显式提供以 conv 开始的 DeepNet op 链")
    names = [op.get("name") for op in ops if op.get("op") in ("conv", "fc")]
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("每个线性 op 必须提供唯一且非空的 name")
    result = []
    linear_index = 0
    for name, bit in zip(SCHEDULE_LAYERS, bits):
        linear = ops[linear_index]
        kind = "fc" if name == "fc1" else "conv"
        if linear.get("name") != name or linear.get("op") != kind:
            raise ValueError(f"schedule 层 {name} 与显式 op 元数据不匹配")
        act_index = linear_index + 1
        if act_index >= len(ops) or ops[act_index].get("op") != "act":
            raise ValueError(f"{name} 后必须显式提供 act")
        next_index = act_index + 1
        pools = []
        while next_index < len(ops) and ops[next_index].get("op") == "pool":
            pools.append(next_index)
            next_index += 1
        if next_index >= len(ops) or ops[next_index].get("op") not in ("conv", "fc"):
            raise ValueError(f"{name} 的 act 后必须是 pool* 与一个线性 op")
        next_op = ops[next_index]
        result.append(ScheduleOp(name, bit, linear_index, act_index, next_index,
                                 next_op["name"], next_op["op"], tuple(pools)))
        linear_index = next_index
    if linear_index != len(ops) - 1:
        raise ValueError("五个激活之后必须恰好有一个显式线性输出头")
    return tuple(result)


def _schedule_depth(layers: tuple[ScheduleOp, ...], extra_depth: int) -> int:
    if type(extra_depth) is not int or extra_depth not in (0, 1):
        raise ValueError("p2_before_conv_extra_depth 只能为 0（旧融合）或 1（im2col_ct）")
    # 对齐 planner.required_depth 与 DeepNet.protocol_ops：线性 1、P2 1、
    # hybrid 重置、pool 融入下一线性、最后输出头 1；独立 im2col 另计。
    depth = maximum = 0
    for layer in layers:
        depth += layer.linear_depth
        maximum = max(maximum, depth)
        if layer.bit:
            depth = 0
        else:
            depth += layer.p2_depth
            if layer.next_linear_op == "conv":
                depth += extra_depth
            maximum = max(maximum, depth)
    return max(maximum, depth + 1)


def schedule_depth(schedule, ops: list[dict], *, p2_before_conv_extra_depth: int = 0) -> int:
    """独立深度检查：默认对齐旧融合路径，显式 im2col_ct 路径须传额外深度 1。"""
    return _schedule_depth(schedule_ops(schedule, ops), p2_before_conv_extra_depth)


def validate_schedule(schedule, ops: list[dict], depth_budget: int = 6, *,
                      p2_before_conv: bool = False,
                      p2_before_conv_extra_depth: int = 0) -> tuple[ScheduleOp, ...]:
    """只供显式调用的 B1 校验，不接入或改变旧 DeepNet/runner。

    先检查完整 op 元数据和深度，再检查 P2-before-conv 能力。开启能力不等于
    集成成功；若拆出 im2col_ct，必须传 p2_before_conv_extra_depth=1。
    """
    if type(depth_budget) is not int or depth_budget < 0:
        raise ValueError("depth_budget 必须是非负整数")
    if type(p2_before_conv) is not bool:
        raise ValueError("p2_before_conv 必须是 bool")
    layers = schedule_ops(schedule, ops)
    need = _schedule_depth(layers, p2_before_conv_extra_depth)
    if need > depth_budget:
        raise DepthBudgetExceeded(f"schedule 所需深度 {need} 超过预算 {depth_budget}")
    for layer in layers:
        if not layer.bit and layer.next_linear_op == "conv" and not p2_before_conv:
            raise P2BeforeConvUnsupported(
                f"{layer.name} 的 P2 后接 {layer.next_linear_name}，独立 B1 接口默认未启用")
    return layers


class Im2colBlocks(list):
    """密文 block 列表；绑定不暴露旋转计数，None 表示未知而非零。"""

    extra_depth = 1
    linear_sparse_calls = 1
    rotations = None


def im2col_ct(ct, in_ch: int, in_h: int, in_w: int, k: int, stride: int,
              pad: int, ctx) -> Im2colBlocks:
    """flatten(CHW) 密文（或连续 blocks）-> tap-major 零 padding im2col。

    使用真实 ctx.linear_sparse(blocks, in_idx, out_idx, values, out_len)，
    Ciphertext.size 是属性、ctx.slot_capacity() 是方法。前面的输入块必须满槽，
    最后一个可为有效长度或满槽；输出按 to_blocks 的 R=slots/windows 排列，
    最后一个块补零到满槽，要求 windows 整除 slots。

    linear_sparse 消耗 1 层，后续 matmul_rows 还要 1 层：比旧融合
    P2 -> conv 路径额外多 1 层，不是零深度排列。rotations 未知。
    绑定拒绝纯零输出块；极端 padding 导致这种块时本接口提前报错，
    不伪造密文或重加密。未集成 P2 系数折叠、pool 或模型执行。
    """
    for name, value in (("in_ch", in_ch), ("in_h", in_h), ("in_w", in_w),
                        ("k", k), ("stride", stride), ("pad", pad)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < (0 if name == "pad" else 1)):
            raise ValueError(f"{name} 必须是{'非负' if name == 'pad' else '正'}整数")
    oh = (in_h + 2 * pad - k) // stride + 1
    ow = (in_w + 2 * pad - k) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("卷积输出空间尺寸必须为正")
    slots = ctx.slot_capacity()
    windows = oh * ow
    if slots <= 0 or slots % windows:
        raise ValueError(f"{windows} windows 必须整除 {slots} slots")
    blocks = list(ct) if isinstance(ct, (list, tuple)) else [ct]
    input_len = in_ch * in_h * in_w
    if len(blocks) != math.ceil(input_len / slots):
        raise ValueError("输入密文 block 数量与 CHW 不匹配")
    for index, block in enumerate(blocks):
        live = min(slots, input_len - index * slots)
        if block.size not in (live, slots):
            raise ValueError("输入密文必须连续按 CHW 满槽分块，末块可补零")
    ii, oi = [], []
    for ci in range(in_ch):
        for ki in range(k):
            for kj in range(k):
                tap = (ci * k + ki) * k + kj
                for oy in range(oh):
                    y = oy * stride + ki - pad
                    for ox in range(ow):
                        x = ox * stride + kj - pad
                        if 0 <= y < in_h and 0 <= x < in_w:
                            ii.append((ci * in_h + y) * in_w + x)
                            oi.append(tap * windows + oy * ow + ox)
    out_blocks = math.ceil(in_ch * k * k * windows / slots)
    if len({index // slots for index in oi}) != out_blocks:
        raise ValueError("linear_sparse 不支持 padding 产生的纯零输出 block")
    result = ctx.linear_sparse(blocks, np.asarray(ii, dtype=np.int64),
                               np.asarray(oi, dtype=np.int64),
                               np.ones(len(ii), dtype=np.float64), out_blocks * slots)
    return Im2colBlocks(result)


def im2col_rows(x: np.ndarray, k: int, stride: int, pad: int) -> tuple[np.ndarray, int, int]:
    """(C,H,W) -> rows (C*k*k, OH*OW) in (c, ki, kj) x (oh, ow) order."""
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = np.empty((c * k * k, oh * ow), dtype=np.float64)
    t = 0
    for ci in range(c):
        for ki in range(k):
            for kj in range(k):
                patch = xp[ci, ki:ki + stride * oh:stride, kj:kj + stride * ow:stride]
                rows[t] = patch.reshape(-1)
                t += 1
    return rows, oh, ow


def to_blocks(rows: np.ndarray, slots: int) -> list[np.ndarray]:
    """Split rows into ciphertext-sized blocks of R = slots / W rows."""
    n_rows, windows = rows.shape
    if slots % windows:
        raise ValueError(f"{windows} windows do not divide {slots} slots "
                         "(pad the spatial size to a power of two)")
    r = slots // windows
    blocks = []
    for b in range(math.ceil(n_rows / r)):
        chunk = np.zeros((r, windows))
        part = rows[b * r:(b + 1) * r]
        chunk[:len(part)] = part
        blocks.append(chunk.reshape(-1))
    return blocks


def conv_reference(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   stride: int, pad: int) -> np.ndarray:
    rows, oh, ow = im2col_rows(x, weight.shape[-1], stride, pad)
    out = weight.reshape(weight.shape[0], -1) @ rows + bias[:, None]
    return out.reshape(weight.shape[0], oh, ow)


# ------------------------------------------------------------------ protocol

def client_encrypt_share(be, ctx, share: np.ndarray, k: int, stride: int,
                         pad: int) -> tuple[list[bytes], int]:
    """Client: im2col its share and encrypt each block (wire profile aware)."""
    rows, oh, ow = im2col_rows(share, k, stride, pad)
    blobs = [be.encrypt_for_send(ctx, blk) if not be.wire.is_baseline
             else be.serialize(be.encrypt(ctx, blk))
             for blk in to_blocks(rows, ctx.slot_capacity())]
    return blobs, oh * ow


def server_conv(be, ctx, blobs: list[bytes], server_share: np.ndarray,
                weight: np.ndarray, bias: np.ndarray, stride: int, pad: int):
    """Server: add its share block-wise, then Z = W_flat @ rows (+ bias)."""
    k = weight.shape[-1]
    rows_s, oh, ow = im2col_rows(server_share, k, stride, pad)
    windows = oh * ow
    blocks_s = to_blocks(rows_s, ctx.slot_capacity())
    cts = []
    for blob, blk in zip(blobs, blocks_s):
        ct = be.deserialize(ctx, blob)
        cts.append(be.add_plain(ct, blk))
    outs = ctx.matmul_rows(cts, weight.reshape(weight.shape[0], -1), windows)
    r = ctx.slot_capacity() // windows
    out = []
    for c, ct in enumerate(outs):
        b = bias[c * r:(c + 1) * r]
        out.append(be.add_plain(ct, np.repeat(b, windows)))
    return out, windows
