#!/usr/bin/env bash
# Build the LitePI SEAL backend (SEAL 3.3.2 is built in-place by SCI's tree).
set -euo pipefail

WZY=/mnt/d/王梓杨/项目组/wzy
SEAL_DIR="$WZY/third_party/EzPC/SCI/extern/SEAL"
PY="${PYTHON:-$WZY/.venv311/bin/python}"

PYBIND_INC=$("$PY" -c 'import pybind11; print(pybind11.get_include())')
PY_INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
EXT=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

cd "$WZY/seal"
g++ -O3 -shared -fPIC -std=c++17 -maes -msse4.1 -mavx2 -mpclmul \
    -I"$PYBIND_INC" -I"$PY_INC" -I"$SEAL_DIR/native/src" \
    litepi_seal.cpp "$SEAL_DIR/native/lib/libseal.a" -lpthread \
    -o "litepi_seal${EXT}"

echo "built $(pwd)/litepi_seal${EXT}"
