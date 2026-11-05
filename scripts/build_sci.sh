#!/usr/bin/env bash
# Build SCI (CrypTFlow2) from EzPC for LitePI MPC backend.
# Run inside WSL2 Ubuntu 22.04.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCI_DIR="${REPO_ROOT}/third_party/EzPC/SCI"
BUILD_DIR="${SCI_DIR}/build"
INSTALL_DIR="${SCI_DIR}/install"

echo "[1/4] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libgmp-dev libmpfr-dev libssl-dev libeigen3-dev \
    python3 python3-pip python3-dev

echo "[2/4] Configuring SCI..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_NETWORKS=OFF \
      ..

echo "[3/4] Building SCI (this takes a few minutes)..."
cmake --build . --target install --parallel"$(nproc)"

echo "[4/4] Verifying build..."
ls -la "${INSTALL_DIR}/lib/"
ls -la "${INSTALL_DIR}/include/" | head -20

echo "SCI build complete. Install prefix: ${INSTALL_DIR}"
echo "Next: build the pybind11 wrapper (scripts/build_litepi_sci.sh)"
