#!/usr/bin/env bash
# Build SCI (CrypTFlow2) from EzPC for LitePI MPC backend.
# Run inside WSL2 Ubuntu 22.04.
set -euo pipefail

WZY_DIR="/mnt/d/王梓杨/项目组/wzy"
SCI_DIR="${WZY_DIR}/third_party/EzPC/SCI"
BUILD_DIR="${SCI_DIR}/build"
INSTALL_DIR="${SCI_DIR}/install"

echo "[1/4] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libgmp-dev libmpfr-dev libssl-dev \
    python3 python3-pip python3-dev

echo "[2/4] Configuring SCI..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DBUILD_TESTS=ON \
      -DBUILD_NETWORKS=OFF \
      ..

echo "[3/4] Building SCI (this takes a few minutes)..."
cmake --build . --target install --parallel"$(nproc)"

echo "[4/4] Verifying build..."
ls -la "${INSTALL_DIR}/lib/"
ls -la "${INSTALL_DIR}/include/" | head -20

echo "SCI build complete. Install prefix: ${INSTALL_DIR}"
echo "Next: build the pybind11 wrapper (scripts/build_litepi_sci.sh)"
