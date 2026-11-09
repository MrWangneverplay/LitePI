#!/usr/bin/env bash
# 使用已有的本地 EzPC/SCI 源码构建并安装 SCI，不自动获取源码。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WZY_DIR="${WZY_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
SCI_DIR="${SCI_DIR:-${WZY_DIR}/third_party/EzPC/SCI}"
SCI_BUILD_DIR="${SCI_BUILD_DIR:-${SCI_DIR}/build}"
SCI_INSTALL_DIR="${SCI_INSTALL_DIR:-${SCI_BUILD_DIR}/install}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-$(nproc 2>/dev/null || printf '1')}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "JOBS 必须为正整数。"
[[ "${INSTALL_DEPS}" == 0 || "${INSTALL_DEPS}" == 1 ]] || fail "INSTALL_DEPS 只能为 0 或 1。"
[[ -f "${SCI_DIR}/CMakeLists.txt" && -f "${SCI_DIR}/src/Math/CMakeLists.txt" ]] || \
    fail "缺少本地 SCI 源码：${SCI_DIR}；请先提供已确认版本的上游源码，本脚本不会克隆或猜测 commit。"
# 本地上游会在 Eigen 源码缺失时自动初始化子模块，提前拒绝该情况。
[[ -f "${SCI_DIR}/extern/eigen/CMakeLists.txt" ]] || \
    fail "缺少本地 Eigen 子模块：${SCI_DIR}/extern/eigen；请先准备依赖源码，不自动下载。"

if [[ "${INSTALL_DEPS}" == 1 ]]; then
    command -v apt-get >/dev/null 2>&1 || fail "INSTALL_DEPS=1 仅支持 apt-get；其他环境请自行安装依赖。"
    command -v sudo >/dev/null 2>&1 || fail "INSTALL_DEPS=1 需要 sudo。"
    sudo apt-get update
    sudo apt-get install -y build-essential cmake pkg-config \
        libgmp-dev libmpfr-dev libssl-dev python3 python3-pip python3-dev
fi

missing=()
for tool in cmake make "${CC:-cc}" "${CXX:-c++}" pkg-config "${PYTHON}"; do
    command -v "${tool}" >/dev/null 2>&1 || missing+=("命令 ${tool}")
done
if command -v pkg-config >/dev/null 2>&1; then
    for package in gmp mpfr openssl; do
        pkg-config --exists "${package}" || missing+=("开发依赖 ${package}（pkg-config 未找到）")
    done
fi
if (( ${#missing[@]} )); then
    printf '缺少依赖：\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    fail "默认不安装依赖；请自行补齐，或在 apt-get 环境显式设置 INSTALL_DEPS=1。"
fi
PYTHON="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
"${PYTHON}" -c 'import subprocess, sys; v = subprocess.check_output(["cmake", "--version"], text=True).split()[2]; sys.exit(0 if tuple(map(int, v.split(".")[:2])) >= (3, 18) else 1)' || \
    fail "需要 CMake >= 3.18，以便与包装模块共用构建工具。"

printf '配置 SCI，Python：%s\n' "${PYTHON}"
cmake -S "${SCI_DIR}" -B "${SCI_BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${SCI_INSTALL_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_NETWORKS=OFF
cmake --build "${SCI_BUILD_DIR}" --parallel "${JOBS}"
cmake --install "${SCI_BUILD_DIR}"

printf 'SCI 安装完成：%s\n' "${SCI_INSTALL_DIR}"
printf '下一步：使用相同的路径覆盖变量运行 scripts/build_litepi_sci.sh。\n'
printf '构建不代表 ARM 支持已验证；原生协议和硬件验证需单独进行。\n'
