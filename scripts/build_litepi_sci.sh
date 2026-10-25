#!/usr/bin/env bash
# Build the litepi_sci pybind11 module (requires SCI built first).
# Run inside WSL2 Ubuntu 22.04.
set -euo pipefail

WZY_DIR="/mnt/d/王梓杨/项目组/wzy"
SCI_DIR="${WZY_DIR}/third_party/EzPC/SCI"
SCI_INSTALL="${SCI_DIR}/install"

if [ ! -d "${SCI_INSTALL}/lib" ]; then
    echo "ERROR: SCI not built. Run scripts/build_sci.sh first."
    exit 1
fi

echo "[1/3] Installing pybind11..."
pip3 install pybind11[global]

echo "[2/3] Building litepi_sci module..."
cd "${WZY_DIR}/sci"
mkdir -p build
cd build
cmake -DSCI_INSTALL_DIR="${SCI_INSTALL}" ..
cmake --build . --parallel"$(nproc)"

echo "[3/3] Installing module to analysis/ ..."
cmake --install .

echo "litepi_sci build complete. Verify with:"
echo "  python3 -c 'import litepi_sci; print(litepi_sci.__doc__)'"
