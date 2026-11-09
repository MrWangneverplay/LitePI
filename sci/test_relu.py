import multiprocessing as mp
import sys
import sysconfig
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
    print(f"litepi_sci 实际路径：{module_path}", flush=True)
    return litepi_sci


litepi_sci = _load_litepi_sci()

PORT = 32000
BW = 32
MASK = (1 << BW) - 1


def signed_val(x, bw):
    x &= (1 << bw) - 1
    return x - (1 << bw) if x >= (1 << (bw - 1)) else x


def party_run(role, shares, q):
    # spawn 会重新加载入口；fork 也需在子进程再次检查模块来源。
    litepi_sci = _load_litepi_sci()
    r = litepi_sci.SecureReLU(role, PORT, "127.0.0.1")
    arr = np.asarray(shares, dtype=np.uint64)
    y = r.relu(arr, BW)
    q.put(np.asarray(y, dtype=np.uint64))


def main():
    # 明文测试值（有符号，两补码 uint64 表示）
    xs = np.array([10, -5, 3, -20, 100, -1, 0, 7, -2147483648, 2147483647],
                  dtype=np.int64)
    x_uint = (xs & MASK).astype(np.uint64)

    n = len(xs)
    np.random.seed(42)
    x_alice = np.random.randint(0, 1 << BW, size=n, dtype=np.uint64)
    x_bob = (x_uint - x_alice) & MASK

    q1 = mp.Queue()
    q2 = mp.Queue()
    p1 = mp.Process(target=party_run, args=(litepi_sci.ALICE, x_alice, q1))
    p2 = mp.Process(target=party_run, args=(litepi_sci.BOB, x_bob, q2))
    p1.start()
    p2.start()
    y_alice = q1.get()
    y_bob = q2.get()
    p1.join()
    p2.join()

    y_uint = (y_alice.astype(np.uint64) + y_bob.astype(np.uint64)) & MASK
    y_signed = np.array([signed_val(int(v), BW) for v in y_uint], dtype=np.int64)
    expected = np.maximum(xs, 0)

    print("input  :", xs.tolist())
    print("output :", y_signed.tolist())
    print("expect :", expected.tolist())
    assert np.array_equal(y_signed, expected), "ReLU mismatch!"
    print("ReLU two-party test PASSED")


if __name__ == "__main__":
    main()
