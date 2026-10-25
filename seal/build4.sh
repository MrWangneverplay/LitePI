#!/usr/bin/env bash
# Build the LitePI SEAL backend against SEAL 4.1.2 (paper implementation section:
# "SEAL 4.x C++ + pybind11").  The 3.3.2 build in ./litepi_seal*.so is kept as a
# fallback; this script writes to ./seal4 so the two never overwrite each other.
set -euo pipefail

WZY="${WZY:-/mnt/d/王梓杨/项目组/wzy}"
SEAL_DIR="${SEAL_DIR:-$HOME/opt/SEAL-4.1.2}"
PY="${PYTHON:-$WZY/.venv311/bin/python}"
# Extension ABI follows the interpreter used for the build, so the 3.10 build
# (system python, carries torch) goes to a separate directory.
OUT_DIR="${OUT_DIR:-seal4}"

PYBIND_INC=$("$PY" -c 'import pybind11; print(pybind11.get_include())')
PY_INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
EXT=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

mkdir -p "$WZY/seal/$OUT_DIR"
cd "$WZY/seal"
g++ -O3 -shared -fPIC -std=c++17 -maes -msse4.1 -mavx2 -mpclmul \
    -DLITEPI_SEAL_VERSION='"4.1.2"' \
    -I"$PYBIND_INC" -I"$PY_INC" \
    -I"$SEAL_DIR/native/src" -I"$SEAL_DIR/build/native/src" \
    litepi_seal.cpp "$SEAL_DIR/build/lib/libseal-4.1.a" -lpthread \
    -o "$OUT_DIR/litepi_seal${EXT}"

echo "built $(pwd)/$OUT_DIR/litepi_seal${EXT}"
"$PY" -c "import sys; sys.path.insert(0, '$WZY/seal/$OUT_DIR'); import litepi_seal; print('import ok:', litepi_seal.SEAL_VERSION)"
