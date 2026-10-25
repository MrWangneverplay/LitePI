"""Backend validation probes for the LitePI HE stack (data recorded for the paper).

Every check writes one JSON file under ``results/seal_backend/`` so the numbers
quoted in the paper come from a re-runnable script instead of a terminal
transcript.

Run inside WSL:
    .venv311/bin/python -u seal/probes.py all

Checks:
    tenseal_drift     TenSEAL auto_rescale gain error per level (paper profile)
    tenseal_semantics TenSEAL slot semantics (mm / im2col / conv / pack)
    seal_functional   litepi_seal operator correctness + level trajectory
    seal_noise        SEAL fresh-encryption and per-rescale noise floor
    seal_drift        SEAL rescale bookkeeping on the same drift test
    seal_rotation_noise  key-switch rotation noise across profiles and scales
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# The 3.3.2 build lives in seal/, the SEAL 4.1.2 build in seal/seal4/.  Both are
# selected here so the same probes can measure either backend; LITEPI_SEAL_TAG
# names the subdirectory under results/seal_backend so results never mix.
SEAL_DIR = Path(os.environ.get("LITEPI_SEAL_DIR", REPO / "seal"))
OUT = REPO / "results" / "seal_backend" / os.environ.get("LITEPI_SEAL_TAG", "")
if str(SEAL_DIR) not in sys.path:
    sys.path.insert(0, str(SEAL_DIR))

N = 8192
BITS = [40, 21, 21, 21, 21, 21, 21, 40]
SCALE_BITS = 21


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "profile": {
            "poly_modulus_degree": N,
            "coeff_mod_bit_sizes": BITS,
            "global_scale_bits": SCALE_BITS,
        },
        "command": " ".join(sys.argv),
    }


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    payload = {"name": name, **_meta(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ------------------------------------------------------------- TenSEAL ----

def _tenseal_ctx(auto_rescale: bool):
    import tenseal as ts

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                     coeff_mod_bit_sizes=BITS)
    ctx.global_scale = 2.0 ** SCALE_BITS
    ctx.auto_rescale = auto_rescale
    ctx.auto_mod_switch = True
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ts, ctx


def tenseal_drift() -> None:
    """TenSEAL rewrites the scale to global_scale after every rescale, so the
    cleartext is multiplied by global_scale / q_i (about 1.174x) per level."""
    import tenseal as ts

    _, ctx = _tenseal_ctx(auto_rescale=True)
    ones = ts.ckks_vector(ctx, [1.0, 1.0, 1.0, 1.0])
    trail = []
    cur = ones
    for i in range(6):
        trail.append({
            "levels_consumed": i,
            "decrypted": np.round(np.asarray(cur.decrypt()), 6).tolist(),
        })
        if i < 5:
            cur = cur.square()
    x = ts.ckks_vector(ctx, [1.0, 1.0, 1.0, 1.0])
    twice = np.asarray((x * 2.0 + 1.0).decrypt())
    _write("tenseal_drift", {
        "backend": "tenseal",
        "tenseal_version": getattr(ts, "__version__", "unknown"),
        "auto_rescale": True,
        "square_ones_trail": trail,
        "expected_square_trail": [1.0] * 6,
        "x2_plus_1": np.round(twice, 6).tolist(),
        "expected_x2_plus_1": 3.0,
        "gain_per_level_estimate": round(float(trail[3]["decrypted"][0]) ** (1.0 / 3), 6)
        if trail[3]["decrypted"][0] > 0 else None,
    })


def tenseal_semantics() -> None:
    """What the SEAL backend must reproduce: exact x @ M, tap-major im2col with
    window j at output slot j, and pack = concatenation."""
    import tenseal as ts

    _, ctx = _tenseal_ctx(auto_rescale=False)
    rng = np.random.default_rng(7)

    # 1. mm is an exact matrix product, including the cyclic wrap region.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mm_cases = {}
    for name, M in {
        "wrap_m01": np.array([[0.0, 7.0], [0, 0], [0, 0], [0, 0]]),
        "wrap_m31": np.array([[0.0, 0.0], [0, 0], [0, 0], [0, 7.0]]),
        "wrap_m10": np.array([[0.0, 0.0], [7, 0], [0, 0], [0, 0]]),
        "general": np.array([[1.0, 10.0], [0, 1], [2, 0], [0, 0]]),
    }.items():
        got = np.asarray(ts.ckks_vector(ctx, x.tolist()).mm(M.tolist()).decrypt())
        mm_cases[name] = {
            "got": np.round(got, 4).tolist(),
            "exact": np.round(x @ M, 4).tolist(),
            "max_abs_err": float(np.max(np.abs(got - x @ M))),
        }

    # 2. im2col layout: per-window padded (A) vs tap-major (B).
    img = rng.normal(0, 1, (8, 8))
    ic, windows = ts.im2col_encoding(ctx, img.tolist(), 3, 3, 2)
    dec = np.asarray(ic.decrypt())
    patches = np.stack([img[r * 2:r * 2 + 3, c * 2:c * 2 + 3].reshape(-1)
                        for r in range(3) for c in range(3)])
    layout_a = np.concatenate(
        [np.concatenate([p, np.zeros(16 - p.size)]) for p in patches])
    layout_b = np.concatenate([patches.T.reshape(-1), np.zeros(16 * 9 - 81)])

    # 3. conv2d_im2col: window j lands on output slot j.
    ker = rng.normal(0, 1, (3, 3))
    cdec = np.asarray(ic.conv2d_im2col(ker.tolist(), windows).decrypt())
    cref = np.stack([p @ ker.reshape(-1) for p in patches])

    # 4. pack_vectors concatenates.
    p0 = ts.ckks_vector(ctx, np.arange(64, dtype=float).tolist())
    p1 = ts.ckks_vector(ctx, (np.arange(64, dtype=float) + 100).tolist())
    pk = np.asarray(ts.CKKSVector.pack_vectors([p0, p1]).decrypt())

    # 5. the real MLP geometry: 64x64 matrix on a 64-slot vector.
    w = rng.normal(0, 0.1, (64, 64))
    xv = rng.normal(0, 1, 64)
    y = np.asarray(ts.ckks_vector(ctx, xv.tolist()).mm(w.tolist()).decrypt())

    _write("tenseal_semantics", {
        "backend": "tenseal",
        "tenseal_version": getattr(ts, "__version__", "unknown"),
        "auto_rescale": False,
        "mm_cases": mm_cases,
        "im2col": {
            "windows": int(windows),
            "slot_count": int(dec.size),
            "err_layout_a_per_window_padded": float(np.max(np.abs(dec - layout_a))),
            "err_layout_b_tap_major": float(np.max(np.abs(dec - layout_b))),
            "layout": "tap-major (padded taps x windows, row-major)",
        },
        "conv": {
            "windows": int(windows),
            "max_abs_err_vs_patch_dot": float(np.max(np.abs(cdec - cref))),
            "output_slot_mapping": "window j -> slot j",
        },
        "pack": {
            "size": int(pk.size),
            "head": np.round(pk[:3], 4).tolist(),
            "mid": np.round(pk[64:67], 4).tolist(),
        },
        "fc64": {
            "max_abs_err": float(np.max(np.abs(y - xv @ w))),
            "size": int(y.size),
        },
    })


# ---------------------------------------------------------------- SEAL ----

def _seal_ctxs(bits=None, scale_bits=None):
    import litepi_seal as S

    client = S.new_context(N, bits or BITS,
                           SCALE_BITS if scale_bits is None else scale_bits)
    server = S.load_public_context(S.public_context_bytes(client))
    return S, client, server


def seal_functional() -> None:
    S, client, server = _seal_ctxs()
    rng = np.random.default_rng(7)
    out: dict = {"backend": "seal", "seal_version": S.SEAL_VERSION}

    x = np.array([1.0, 2.0, 3.0, 4.0])
    ct = client.encrypt(x)
    out["levels"] = {"fresh_index": int(client.index(ct)),
                     "fresh_scale": float(client.scale(ct)),
                     "chain_length": int(client.index(ct)) + 1}

    M = np.array([[1.0, 10.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.0]])
    mout = client.decrypt(server.mm(ct, M))
    out["mm"] = {
        "got": np.round(mout, 5).tolist(),
        "exact": (x @ M).tolist(),
        "max_abs_err": float(np.max(np.abs(mout - x @ M))),
        "level_after": int(server.index(server.mm(ct, M))),
    }
    # square matrix case (all real MLP layers are wide, add one square case)
    xw = rng.normal(0, 1, 64)
    w = rng.normal(0, 0.1, (64, 64))
    ctw = client.encrypt(xw)
    yw = client.decrypt(server.mm(ctw, w))
    out["mm64"] = {"max_abs_err": float(np.max(np.abs(yw - xw @ w))),
                   "max_abs_value": float(np.max(np.abs(xw @ w)))}

    # im2col + conv2d on an 8x8 image with a 3x3 kernel, stride 2
    img = rng.normal(0, 1, (8, 8))
    ker = rng.normal(0, 1, (3, 3))
    ic, windows = client.im2col_encrypt(img, 3, 3, 2)
    patches = np.stack([img[r * 2:r * 2 + 3, c * 2:c * 2 + 3].reshape(-1)
                        for r in range(3) for c in range(3)])
    padded_taps = ic.size // windows
    layout = np.zeros((padded_taps, windows))
    layout[:9] = patches.T
    icdec = client.decrypt(ic)
    conv = server.conv2d_im2col(ic, ker, windows)
    cdec = client.decrypt(conv)
    cref = patches @ ker.reshape(-1)
    out["im2col_conv"] = {
        "windows": int(windows),
        "im2col_slots": int(ic.size),
        "im2col_windows_padded_taps": int(padded_taps),
        "im2col_max_abs_err_vs_tap_major": float(np.max(np.abs(icdec - layout.reshape(-1)))),
        "im2col_layout": "tap-major (padded taps x windows, row-major)",
        "conv_slots": int(conv.size),
        "got": np.round(cdec, 5).tolist(),
        "exact": np.round(cref, 5).tolist(),
        "max_abs_err": float(np.max(np.abs(cdec - cref))),
        "level_after": int(server.index(conv)),
        "scale_after": float(server.scale(conv)),
        "rescale": "deferred to pack_vectors (rotation at the 2^42 product scale)",
    }

    # pack two 64-slot channels
    ch0 = client.encrypt(np.arange(64, dtype=float))
    ch1 = client.encrypt(np.arange(64, dtype=float) + 100)
    pk = client.decrypt(server.pack_vectors([ch0, ch1]))
    ref = np.concatenate([np.arange(64.0), np.arange(64.0) + 100])
    out["pack"] = {
        "slots": int(pk.size),
        "head": np.round(pk[:3], 5).tolist(),
        "mid": np.round(pk[64:67], 5).tolist(),
        "max_abs_err": float(np.max(np.abs(pk - ref))),
    }

    out["residual_source"] = (
        "mm / conv residuals are dominated by the plaintext factor and by the "
        "fresh-encryption noise, not by the rotation key switch: the operators "
        "multiply before rotating, so the rotation runs at the 2^42 product "
        "scale.  conv defers its rescale to pack_vectors, so the canonical "
        "conv+pack chain also rotates at 2^42 (4e-3 residual).  Packing fresh "
        "2^21 ciphertexts instead stays at the 2.6e-1 key-switch floor; that "
        "case is what seal_rotation_noise.json quantifies."
    )

    # scalar/plain algebra and the level trajectory of the P2 polynomial
    a = client.encrypt(np.array([0.3, -0.7, 2.0, 0.0]))
    b = server.add_plain(a, np.array([0.5, 0.5, 0.5, 0.5]))
    c = server.mul_plain_scalar(b, 2.0)
    sq = server.square(c)
    out["algebra"] = {
        "add_plain": {"got": np.round(client.decrypt(b), 5).tolist(),
                      "exact": [0.8, -0.2, 2.5, 0.5],
                      "level": int(server.index(b))},
        "mul_plain": {"got": np.round(client.decrypt(c), 5).tolist(),
                      "exact": [1.6, -0.4, 5.0, 1.0],
                      "level": int(server.index(c))},
        "square": {"got": np.round(client.decrypt(sq), 5).tolist(),
                   "exact": [2.56, 0.16, 25.0, 1.0],
                   "level": int(server.index(sq))},
    }
    # P2 activation (protocol SSOT section 3.1) and its level cost
    x4 = np.array([0.3, -0.7, 2.0, 0.0])
    a = client.encrypt(x4)
    p2 = server.poly2_eval(a, 0.0625, 0.5, 0.5)
    p2d = client.decrypt(p2)
    p2_exact = 0.0625 + 0.5 * x4 + 0.5 * x4 * x4
    sq_only = server.poly2_eval(a, 0.0, 0.0, 1.0)
    out["p2_poly"] = {
        "got": np.round(p2d, 6).tolist(),
        "exact": np.round(p2_exact, 6).tolist(),
        "max_abs_err": float(np.max(np.abs(p2d - p2_exact))),
        "level_fresh": int(client.index(a)),
        "level_after": int(server.index(p2)),
        "levels_consumed": int(client.index(a)) - int(server.index(p2)),
        "square_only_level_after": int(server.index(sq_only)),
        "square_only_levels": int(client.index(a)) - int(server.index(sq_only)),
        "square_only_max_abs_err": float(
            np.max(np.abs(client.decrypt(sq_only) - x4 * x4))),
    }
    # the protocol's one-level form used by the canonical networks:
    # P2(z) = a2 (z + a1/2a2)^2 + d, with a2/d folded into the next linear layer
    shift = 0.5 / (2.0 * 0.5)
    p2s = server.square(server.add_plain_scalar(a, shift))
    p2s_d = client.decrypt(p2s)
    out["p2_shifted"] = {
        "shift": shift,
        "got": np.round(p2s_d, 6).tolist(),
        "exact": np.round((x4 + shift) ** 2, 6).tolist(),
        "max_abs_err": float(np.max(np.abs(p2s_d - (x4 + shift) ** 2))),
        "levels_consumed": int(client.index(a)) - int(server.index(p2s)),
    }

    # step-1 primitive: a fresh Enc_pk(r) that SEAL accepts as an addend of a
    # ciphertext already at a lower level (level and scale both inherited)
    fresh_idx = int(client.index(ct))
    mm_ct = server.mm(ct, M)
    r = np.array([0.1, -0.2])
    like = server.encrypt_like(r, mm_ct, True)
    summed = server.add_cipher(mm_ct, like)
    sd = client.decrypt(summed)
    out["encrypt_like"] = {
        "fresh_level": fresh_idx,
        "target_level": int(server.index(mm_ct)),
        "encrypted_level": int(server.index(like)),
        "scale_target": float(server.scale(mm_ct)),
        "scale_encrypted": float(server.scale(like)),
        "add_plain_share_max_abs_err": float(np.max(np.abs(sd - (mout + r)))),
        "level_after_add": int(server.index(summed)),
    }

    # wire format
    blob = conv.serialize()
    back = server.deserialize(blob)
    out["wire"] = {"serialized_bytes": len(blob),
                   "roundtrip_err": float(np.max(np.abs(client.decrypt(back) - cdec))),
                   "server_can_decrypt": None}
    try:
        server.decrypt(conv)
        out["wire"]["server_can_decrypt"] = True
    except Exception as exc:
        out["wire"]["server_can_decrypt"] = type(exc).__name__

    _write("seal_functional", out)


def seal_noise() -> None:
    """Noise floor of the SEAL backend: fresh pk-encryption, one key-switch
    rotation, and the growth per rescale."""
    S, client, server = _seal_ctxs()
    rng = np.random.default_rng(3)
    out: dict = {"backend": "seal", "seal_version": S.SEAL_VERSION}

    cases = {
        "ones": np.ones(64),
        "arange": np.arange(64, dtype=float),
        "arange_x1e-2": np.arange(64, dtype=float) * 1e-2,
        "randn": rng.normal(0, 1, 64),
        "randn_x1e4": rng.normal(0, 1, 64) * 1e4,
    }
    out["fresh"] = {}
    for name, v in cases.items():
        ct = client.encrypt(v)
        d = client.decrypt(ct)
        out["fresh"][name] = {"max_abs_err": float(np.max(np.abs(d - v))),
                              "max_abs_value": float(np.max(np.abs(v)))}

    x = np.arange(64, dtype=float)
    ct = client.encrypt(x)
    out["rotation"] = {}
    full = np.zeros(int(client.slot_capacity()))
    full[:64] = x
    for steps in (1, 5, 64, 1200, 4064):
        rot = server.rotate(ct, steps)
        got = client.decrypt(rot)
        ref = np.roll(full, -steps)[:64]
        out["rotation"][str(steps)] = {
            "max_abs_err": float(np.max(np.abs(got - ref))),
            "level": int(server.index(rot)),
        }

    acc = ct
    out["rescale_growth"] = []
    for i in range(1, 5):
        acc = server.mul_plain(acc, np.ones(64))
        d = client.decrypt(acc)
        out["rescale_growth"].append({
            "rescales": i,
            "level": int(server.index(acc)),
            "max_abs_err": float(np.max(np.abs(d - x))),
        })

    _write("seal_noise", out)


def seal_drift() -> None:
    """Same drift test as TenSEAL, but on SEAL's exact rescale bookkeeping."""
    S, client, server = _seal_ctxs()
    ones = client.encrypt(np.ones(4))
    trail = []
    cur = ones
    for i in range(6):
        trail.append({"levels_consumed": i,
                      "decrypted": np.round(client.decrypt(cur), 6).tolist()})
        if i < 5:
            cur = server.square(cur)
    x = client.encrypt(np.ones(4))
    twice = client.decrypt(server.add_plain(server.mul_plain_scalar(x, 2.0), [1.0] * 4))
    _write("seal_drift", {
        "backend": "seal",
        "seal_version": S.SEAL_VERSION,
        "square_ones_trail": trail,
        "expected_square_trail": [1.0] * 6,
        "x2_plus_1": np.round(twice, 6).tolist(),
        "expected_x2_plus_1": 3.0,
    })


def _rotation_block(client, server, source, steps):
    """Rotate `source` (a plaintext ring, length = slot capacity) and report the
    absolute error per step.  `noise_coeff_units` is the error rescaled back to
    coefficient units (err * ciphertext scale), which is the quantity that is
    independent of the encoding scale."""
    n2 = int(client.slot_capacity())
    ct = client.encrypt(source)
    out = {
        "fresh_level": int(client.index(ct)),
        "fresh_scale": float(client.scale(ct)),
        "fresh_max_abs_err": float(np.max(np.abs(client.decrypt(ct) - source))),
        "rotations": {},
    }
    for steps_i in steps:
        rot = server.rotate(ct, steps_i)
        got = client.decrypt(rot)
        err = np.abs(got - np.roll(source, -steps_i))
        top = np.argsort(err)[-3:][::-1]
        scale = float(server.scale(rot))
        out["rotations"][str(steps_i)] = {
            "max_abs_err": float(err.max()),
            "argmax_slot": int(err.argmax()),
            "top_slots": [[int(i), float(got[i])] for i in top],
            "level": int(server.index(rot)),
            "scale": scale,
            "noise_coeff_units": float(err.max() * scale),
        }
    return out


def seal_rotation_noise() -> None:
    """Rotation key-switch noise at the paper profile.

    A rotation is a key switch, and the key-switch noise is a fixed number of
    coefficient units (2^18.5-2^20.7 here), so the absolute error it causes is
    that noise divided by whatever scale the ciphertext has when it is rotated.
    The operators therefore multiply first (scale 2^42) and rotate the product;
    only `pack` still rotates a fresh ciphertext at 2^21.  The check records
    (a) three parameter profiles, (b) error vs encoding scale at the paper
    profile, (c) error at every remaining level, and (d) the same rotation on a
    product, which is what mm/conv actually do.
    """
    steps = (0, 1, 64, 100, 1200, 4064)
    profiles = {
        "paper_8prime_21bit": (BITS, SCALE_BITS),
        "seal_example_5x40": ([40, 40, 40, 40, 40], 40),
        "six_30bit": ([30, 30, 30, 30, 30, 30], 30),
    }
    out: dict = {"backend": "seal", "rotation_steps": list(steps), "profiles": {}}
    n2 = N // 2
    zeros = np.zeros(n2)
    for name, (bits, scale_bits) in profiles.items():
        S, client, server = _seal_ctxs(bits, scale_bits)
        block = _rotation_block(client, server, zeros, steps)
        block["coeff_mod_bit_sizes"] = list(bits)
        block["global_scale_bits"] = scale_bits
        block["chain_length"] = int(client.index(client.encrypt(zeros))) + 1
        out["profiles"][name] = block
        out.setdefault("seal_version", S.SEAL_VERSION)
        del client, server

    # (b) error vs encoding scale at the fixed paper parameter set
    S, client, server = _seal_ctxs()
    out["paper_profile_rotation_vs_scale"] = []
    for scale_bits in (21, 26, 31, 36, 41, 46, 51):
        ct = client.encrypt(zeros, 2.0 ** scale_bits)
        rot = server.rotate(ct, 100)
        err = float(np.max(np.abs(client.decrypt(rot))))
        out["paper_profile_rotation_vs_scale"].append({
            "encoding_scale_bits": scale_bits,
            "scale": float(server.scale(rot)),
            "max_abs_err": err,
            "noise_coeff_units": err * float(server.scale(rot)),
        })

    # (c) rotation error at every level of the paper chain, with the scale drift
    #     that `mul_plain` produces when the factor is encoded at the current
    #     (already inflated) ciphertext scale
    x = np.arange(64, dtype=float)
    ring = np.zeros(n2)
    ring[:64] = x
    cur = client.encrypt(x)
    sweep = []
    for i in range(6):
        rot = server.rotate(cur, 100)
        got = client.decrypt(rot)
        ref = np.roll(ring, -100)[:64]
        err = float(np.max(np.abs(got - ref)))
        scale = float(server.scale(rot))
        sweep.append({
            "step": i,
            "level": int(server.index(rot)),
            "scale": scale,
            "max_abs_err": err,
            "noise_coeff_units": err * scale,
        })
        if i < 5:
            cur = server.mul_plain(cur, np.ones(64))
    out["paper_profile_rotation_vs_level"] = sweep

    # (d) rotation on a product: multiply_plain lifts the scale to 2^42, which is
    #     the order the mm/conv operators rotate in, so the same key-switch noise
    #     is measured against a 2^42 scale instead of 2^21
    prod = server.mul_plain(client.encrypt(x), np.ones(64))
    rot_prod = server.rotate(prod, 100)
    got = client.decrypt(rot_prod)
    ref = np.roll(ring, -100)[:64]
    err = float(np.max(np.abs(got - ref)))
    scale = float(server.scale(rot_prod))
    out["paper_profile_rotation_on_product"] = {
        "note": "multiply_plain by 1.0 first, then rotate by 100",
        "level": int(server.index(rot_prod)),
        "scale": scale,
        "max_abs_err": err,
        "noise_coeff_units": err * scale,
        "improvement_vs_fresh_scale": (
            float(out["paper_profile_rotation_vs_level"][0]["max_abs_err"]) / err
            if err > 0 else None
        ),
    }

    out["model"] = {
        "rule": "absolute decrypt error = key-switch noise (coefficient units) / ciphertext scale",
        "fresh_encryption_noise_units": "about 2^13 (measured 7.7e3 at 2^21)",
        "key_switch_noise_units": "2^18.5-2^20.7 for every coefficient modulus tried",
        "implication": (
            "the key-switch noise is scale independent, so the operators multiply "
            "before rotating: at the 2^42 product scale the same rotation is ~7e-8 "
            "instead of ~0.2. conv defers its rescale so the canonical pack also "
            "rotates at 2^42 and lands at ~4e-3; packing fresh 2^21 ciphertexts "
            "stays at the 2.6e-1 key-switch floor"
        ),
        "tenseal_reference_same_profile": (
            "TenSEAL 0.3.18 at the identical parameters measures 1.2e-2 on the same "
            "4x2 mm and 1.7e-3 on the 64x64 layer; see tenseal_semantics.json"
        ),
        "operator_errors_see": "seal_functional.json (mm, mm64, conv, pack)",
    }
    _write("seal_rotation_noise", out)


CHECKS = {
    "tenseal_drift": tenseal_drift,
    "tenseal_semantics": tenseal_semantics,
    "seal_functional": seal_functional,
    "seal_noise": seal_noise,
    "seal_drift": seal_drift,
    "seal_rotation_noise": seal_rotation_noise,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(CHECKS) if which == "all" else [which]
    for name in names:
        print(f"\n=== {name} ===")
        CHECKS[name]()


if __name__ == "__main__":
    main()
