#!/usr/bin/env bash
# 统一委托 CMake 包装模块入口，不再维护手工 g++ 编译链。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WZY_DIR="${WZY_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
exec bash "${WZY_DIR}/scripts/build_litepi_sci.sh" "$@"
