#!/usr/bin/env bash
# 本地 WSL 重编 litepi_sci：固定使用仓库内 .venv311 (Python 3.11)。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON="${PYTHON:-$PWD/.venv311/bin/python3.11}"
export JOBS="${JOBS:-$(nproc)}"
exec bash scripts/build_litepi_sci.sh
