#!/usr/bin/env bash
# Build SEAL 4.1.2 WITH its zlib/zstd codecs, then the litepi_seal extension.
#
# Why: the E4 wire profiles "zstd" and "compact" need SEAL's serialisation
# codecs.  The SEAL build used for every byte figure recorded up to
# 2026-10-29 had no codec (a fresh ciphertext serialised to 917,625 B; the
# same object is 437,668 B with zstd).  litepi_seal now passes the codec
# explicitly, so "baseline" stays byte-identical on either build; only the
# compressed profiles need this build.
#
#   WZY=/mnt/d/王梓杨/项目组/wzy PYTHON=$WZY/.venv311/bin/python bash seal/build_seal4_deps.sh
#
# SEAL_BUILD_DEPS=ON downloads zlib, zstd and Microsoft GSL from GitHub at
# configure time; the codecs are linked into libseal-4.1.a.
set -euo pipefail

WZY="${WZY:-/mnt/d/王梓杨/项目组/wzy}"
SEAL_DIR="${SEAL_DIR:-$HOME/opt/SEAL-4.1.2-codecs}"
PY="${PYTHON:-$WZY/.venv311/bin/python}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$SEAL_DIR" ]; then
    git clone --depth 1 --branch v4.1.2 https://github.com/microsoft/SEAL.git "$SEAL_DIR"
fi
cmake -S "$SEAL_DIR" -B "$SEAL_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DSEAL_BUILD_DEPS=ON -DSEAL_USE_ZSTD=ON -DSEAL_USE_ZLIB=ON \
    -DSEAL_USE_MSGSL=ON -DSEAL_USE_INTEL_HEXL=OFF
cmake --build "$SEAL_DIR/build" -j "$JOBS"

PYBIND_INC=$("$PY" -c 'import pybind11; print(pybind11.get_include())')
PY_INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
EXT=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
OUT_DIR="${OUT_DIR:-seal4}"
mkdir -p "$WZY/seal/$OUT_DIR"
cd "$WZY/seal"
g++ -O3 -shared -fPIC -std=c++17 \
    -DLITEPI_SEAL_VERSION='"4.1.2"' \
    -I"$PYBIND_INC" -I"$PY_INC" \
    -I"$SEAL_DIR/native/src" -I"$SEAL_DIR/build/native/src" \
    -I"$SEAL_DIR/build/thirdparty/msgsl-src/include" \
    litepi_seal.cpp "$SEAL_DIR/build/lib/libseal-4.1.a" -lpthread \
    -o "$OUT_DIR/litepi_seal${EXT}"

"$PY" - <<EOF
import sys
sys.path.insert(0, "$WZY/seal/$OUT_DIR")
import litepi_seal
print("import ok:", litepi_seal.SEAL_VERSION, "codecs:", litepi_seal.compr_modes())
EOF
