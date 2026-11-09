#!/usr/bin/env bash
# 用指定 Python 的 pybind11 构建包装模块，要求 SCI 已安装。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WZY_DIR="${WZY_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
SCI_DIR="${SCI_DIR:-${WZY_DIR}/third_party/EzPC/SCI}"
SCI_BUILD_DIR="${SCI_BUILD_DIR:-${SCI_DIR}/build}"
SCI_INSTALL_DIR="${SCI_INSTALL_DIR:-${SCI_BUILD_DIR}/install}"
SCI_PACKAGE_DIR="${SCI_INSTALL_DIR}/lib/cmake/SCI"
# Eigen 由上游在源码树配置，不相对可能被覆盖的 SCI_BUILD_DIR 猜测。
EIGEN3_DIR="${EIGEN3_DIR:-${SCI_DIR}/extern/eigen/build}"
LITEPI_BUILD_DIR="${WZY_DIR}/sci/build"
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
# Math-only 加载不进入带 unused HE dependency 的 SCIConfig；检查实际使用的安装文件。
for module in SCITargets.cmake source_of_randomness.cmake FindGMP.cmake; do
    [[ -f "${SCI_PACKAGE_DIR}/${module}" ]] || \
        fail "SCI 安装缺少 ${SCI_PACKAGE_DIR}/${module}；请提供完整且兼容的 Math-only build。"
done
target_configs=("${SCI_PACKAGE_DIR}"/SCITargets-*.cmake)
[[ -f "${target_configs[0]}" ]] || \
    fail "SCI 安装缺少 ${SCI_PACKAGE_DIR}/SCITargets-*.cmake；请提供完整且兼容的 Math-only build。"
[[ -f "${SCI_INSTALL_DIR}/lib/libSCI-Math.a" && \
   -f "${SCI_INSTALL_DIR}/include/Math/math-functions.h" ]] || \
    fail "SCI 安装不完整：${SCI_INSTALL_DIR}；请使用相同的 SCI_DIR、SCI_BUILD_DIR、SCI_INSTALL_DIR 提供兼容的 Math-only build。"
for config in Eigen3Config.cmake Eigen3Targets.cmake; do
    [[ -f "${EIGEN3_DIR}/${config}" ]] || \
        fail "缺少本地 Eigen 配置：${EIGEN3_DIR}/${config}；请设置 EIGEN3_DIR，不自动下载或配置 Eigen。"
done

if [[ "${INSTALL_DEPS}" == 1 ]]; then
    command -v apt-get >/dev/null 2>&1 || fail "INSTALL_DEPS=1 仅支持 apt-get；其他环境请自行安装依赖。"
    command -v sudo >/dev/null 2>&1 || fail "INSTALL_DEPS=1 需要 sudo。"
    sudo apt-get update
    sudo apt-get install -y build-essential cmake pkg-config \
        libgmp-dev libmpfr-dev libssl-dev python3 python3-pip python3-dev
    command -v "${PYTHON}" >/dev/null 2>&1 || fail "找不到指定 Python：${PYTHON}。"
    "${PYTHON}" -m pip install pybind11
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
    fail "需要 CMake >= 3.18。"
"${PYTHON}" -c 'import os, sys, sysconfig; sys.exit(0 if os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")) else 1)' || \
    fail "指定 Python 缺少开发头文件：${PYTHON}；请安装与该解释器匹配的开发包。"
PYBIND11_DIR="$("${PYTHON}" -m pybind11 --cmakedir)" || \
    fail "指定 Python 找不到 pybind11；请为 ${PYTHON} 安装 pybind11，或显式设置 INSTALL_DEPS=1。"
[[ -f "${PYBIND11_DIR}/pybind11Config.cmake" ]] || fail "pybind11 CMake 配置不存在：${PYBIND11_DIR}。"

printf '配置 litepi_sci，Python：%s\n' "${PYTHON}"
cmake -S "${WZY_DIR}/sci" -B "${LITEPI_BUILD_DIR}" \
    -DSCI_SOURCE_DIR:PATH="${SCI_DIR}" \
    -DSCI_BUILD_DIR:PATH="${SCI_BUILD_DIR}" \
    -DSCI_INSTALL_DIR:PATH="${SCI_INSTALL_DIR}" \
    -DSCI_DIR:PATH="${SCI_PACKAGE_DIR}" \
    -DEIGEN3_DIR:PATH="${EIGEN3_DIR}" \
    -DPython_EXECUTABLE:FILEPATH="${PYTHON}" \
    -DPYBIND11_FINDPYTHON=ON \
    -Dpybind11_DIR:PATH="${PYBIND11_DIR}"
cmake --build "${LITEPI_BUILD_DIR}" --parallel "${JOBS}"
cmake --install "${LITEPI_BUILD_DIR}"

printf 'litepi_sci 已安装到 %s/analysis，目标解释器：%s\n' "${WZY_DIR}" "${PYTHON}"
printf '未执行导入、密码 loopback 或硬件测试；构建不代表 ARM 支持已验证。\n'
