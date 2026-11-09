import sys
import sysconfig
from pathlib import Path

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
print("doc:", litepi_sci.__doc__)
print("ALICE =", litepi_sci.ALICE, "BOB =", litepi_sci.BOB)
