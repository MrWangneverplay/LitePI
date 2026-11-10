import multiprocessing as mp
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np

ANALYSIS_DIR = (Path(__file__).resolve().parents[1] / "analysis").resolve()


def _load_litepi_sci():
    sys.path.insert(0, str(ANALYSIS_DIR))
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise ImportError("当前 Python 未提供扩展 ABI 后缀。")
    binary = ANALYSIS_DIR / f"litepi_sci{suffix}"
    if not binary.is_file():
        raise ImportError(f"缺少 analysis 中当前 Python ABI 的模块：{binary}；拒绝回退到旧 binary。")
    import litepi_sci
    module_path = Path(litepi_sci.__file__).resolve()
    assert module_path.parent == ANALYSIS_DIR, f"加载了非 analysis 模块：{module_path}"
    assert module_path == binary.resolve(), f"加载了非当前 ABI 模块：{module_path}"
    from sci_relu import require_real_bridge
    require_real_bridge(litepi_sci)
    print(f"litepi_sci 实际路径：{module_path}", flush=True)
    return litepi_sci


litepi_sci = _load_litepi_sci()

PORT = 33000
F = 8
BW = 44  # SCI 使用完整 44 位环，而非 14 位比较窗口。
MASK = (1 << BW) - 1


def signed(v, bw):
    v &= (1 << bw) - 1
    return v - (1 << bw) if v >= (1 << (bw - 1)) else v


def quantize(shares, f, bw):
    scale = 1 << f
    return np.asarray([int(round(s * scale)) & ((1 << bw) - 1)
                       for s in shares], dtype=np.uint64)


def ring_to_float(vals, f, bw):
    scale = 1 << f
    return np.asarray([signed(v, bw) / scale for v in vals], dtype=np.float64)


def party(role, shares_ring, q):
    # spawn 会重新加载入口；fork 也需在子进程再次检查模块来源。
    litepi_sci = _load_litepi_sci()
    r = litepi_sci.SecureReLU(role, PORT, "127.0.0.1")
    t0 = time.perf_counter()
    # Test the deployed real-additive bridge, not just modular reconstruction.
    out = np.asarray(r.relu_real(shares_ring, BW, F), dtype=np.float64)
    dt = time.perf_counter() - t0
    q.put((out, dt, r.bytes_sent()))


def main():
    n = 128
    np.random.seed(7)
    # F=8、14 位有符号定点窗口是 [-32, 32)，不是 [-64, 64)。
    # 保留 [-40, 40) 采样（部分值超出 14 位窗口），本测试走完整 44 位 SCI ReLU。
    z = np.random.uniform(-40, 40, size=n)
    z = np.round(z * (1 << F)) / (1 << F)
    z[:8] = np.asarray([-(1 << (BW - 1)), -33 * (1 << F), -1, 0,
                        1, 33 * (1 << F), (1 << (BW - 1)) - 2,
                        (1 << (BW - 1)) - 1], dtype=np.float64) / (1 << F)

    z_ring = quantize(z, F, BW)
    s_alice = np.random.randint(0, 1 << BW, size=n, dtype=np.uint64)
    s_bob = (z_ring - s_alice) & MASK

    q1 = mp.Queue()
    q2 = mp.Queue()
    p1 = mp.Process(target=party, args=(litepi_sci.ALICE, s_alice, q1))
    p2 = mp.Process(target=party, args=(litepi_sci.BOB, s_bob, q2))
    p1.start()
    p2.start()
    out_a, dt_a, bytes_a = q1.get()
    out_b, dt_b, bytes_b = q2.get()
    p1.join()
    p2.join()

    # No mod 2^BW correction is allowed after the bridge: CKKS adds reals.
    y = out_a + out_b
    expected = np.maximum(z, 0)

    err = np.max(np.abs(y - expected))
    tol = 1.0 / (1 << F)
    ok = err <= tol

    print(f"batch SCI ReLU n={n} bw={BW}: {'PASS' if ok else 'FAIL'}")
    print(f"  max_abs_err={err:.6f} tol={tol:.6f}")
    print(f"  alice: {dt_a*1000:.1f} ms  {bytes_a} B sent")
    print(f"  bob:   {dt_b*1000:.1f} ms  {bytes_b} B sent")
    print(f"  total comm = {bytes_a + bytes_b} B for {n} ReLUs "
          f"= {(bytes_a + bytes_b)/n:.1f} B/elem")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
