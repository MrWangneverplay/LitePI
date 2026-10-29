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

Scope (实验方案/PPT P1: "先做 conv2/conv3 一个种子"): convolutions preceded by
a hybrid layer.  A P2 layer before a convolution would need the im2col
re-layout on ciphertext (rotations, GAZELLE-style) and is not implemented;
planner schedules for deep nets must therefore put a hybrid bit before every
conv after the first one if they are to run on ciphertext.

Layout: a ciphertext holds R rows of W = windows slots; row t is im2col tap
t = (c_in, ki, kj) over all output positions; the output has one row per
output channel, i.e. PyTorch's (C, H, W) flatten order.
"""

from __future__ import annotations

import math

import numpy as np


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
