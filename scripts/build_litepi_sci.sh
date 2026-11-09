#!/usr/bin/env bash
# Build the litepi_sci pybind11 module (requires SCI built first).
# Run inside WSL2 Ubuntu 22.04 with Python 3.11 on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCI_INSTALL="${REPO_ROOT}/third_party/EzPC/SCI/install"
PYTHON="${PYTHON:-python3.11}"

if [ ! -d "${SCI_INSTALL}/lib" ]; then
    echo "ERROR: SCI not built. Run scripts/build_sci.sh first."
    exit 1
fi

if ! "${PYTHON}" -c "import sys; assert sys.version_info[:2] == (3, 11)" 2>/dev/null; then
    echo "ERROR: ${PYTHON} is not Python 3.11 (the protocol profile requires it)."
    exit 1
fi

echo "[1/3] Installing pybind11..."
"${PYTHON}" -m pip install pybind11

echo "[2/3] Building litepi_sci module..."
cd "${REPO_ROOT}/sci"
mkdir -p build
cd build
cmake -DSCI_INSTALL_DIR="${SCI_INSTALL}" \
      -Dpybind11_DIR="$("${PYTHON}" -m pybind11 --cmakedir)" \
      ..
cmake --build . --parallel"$(nproc)"

echo "[3/3] Done (module written to ${REPO_ROOT}/sci). Verify with:"
echo "  ${PYTHON} -c 'import sys; sys.path.insert(0, \"${REPO_ROOT}/sci\"); import litepi_sci; print(litepi_sci.__doc__)'"
