cmake_minimum_required(VERSION 3.18)
# SCI 安装包的随机源模块使用 C 文件探测，需同时启用 C。
project(litepi_sci LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 与构建脚本一致；源码、构建、安装及依赖配置目录均可覆盖。
set(SCI_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/EzPC/SCI"
    CACHE PATH "SCI 本地源码目录")
set(SCI_BUILD_DIR "${SCI_SOURCE_DIR}/build"
    CACHE PATH "SCI 构建目录")
set(SCI_INSTALL_DIR "${SCI_BUILD_DIR}/install"
    CACHE PATH "SCI 安装前缀")
set(SCI_DIR "${SCI_INSTALL_DIR}/lib/cmake/SCI"
    CACHE PATH "SCI 已安装的 CMake 导出目录")

# 上游配置的默认 Eigen 路径相对 build；自定义 build 时优先已有源码配置。
set(_eigen3_default "${SCI_BUILD_DIR}/../extern/eigen/build")
if(EXISTS "${SCI_SOURCE_DIR}/extern/eigen/build/Eigen3Config.cmake")
    set(_eigen3_default "${SCI_SOURCE_DIR}/extern/eigen/build")
endif()
set(EIGEN3_DIR "${_eigen3_default}" CACHE PATH "本地 Eigen3 CMake 配置目录")
foreach(_eigen_file Eigen3Config.cmake Eigen3Targets.cmake)
    if(NOT EXISTS "${EIGEN3_DIR}/${_eigen_file}")
        message(FATAL_ERROR "缺少本地 Eigen 配置：${EIGEN3_DIR}/${_eigen_file}；请设置 EIGEN3_DIR，不自动下载或配置 Eigen。")
    endif()
endforeach()

# 仅绕开所检查的本地 SCI 包中 unused HE dependency：LinearHE 已关闭，
# SCIConfig 却仍无条件查找 SEAL。这不是对所有 SCI 版本的通用兼容承诺。
# 直接加载真实安装导出，保留上游传递依赖及平台 compileflags，不伪造 SEAL target。
foreach(_sci_file SCITargets.cmake source_of_randomness.cmake FindGMP.cmake)
    if(NOT EXISTS "${SCI_DIR}/${_sci_file}")
        message(FATAL_ERROR "SCI 安装缺少 ${SCI_DIR}/${_sci_file}；需要完整且兼容的 Math-only build。")
    endif()
endforeach()
file(GLOB _sci_target_configs "${SCI_DIR}/SCITargets-*.cmake")
if(NOT _sci_target_configs)
    message(FATAL_ERROR "SCI 安装缺少 SCITargets-*.cmake；需要完整且兼容的 Math-only build。")
endif()
foreach(_sci_targets_file "${SCI_DIR}/SCITargets.cmake" ${_sci_target_configs})
    file(READ "${_sci_targets_file}" _sci_targets_content)
    if(_sci_targets_content MATCHES "LinearHE|SCI-HE|(^|[^A-Za-z0-9_])([Ss][Ee][Aa][Ll]|lib[Ss][Ee][Aa][Ll])([^A-Za-z0-9_]|$)")
        message(FATAL_ERROR "${_sci_targets_file} 含非 Math-only 的 LinearHE/SEAL 目标或依赖；请提供兼容的 Math-only build，不以伪造 SEAL target 绕过。")
    endif()
endforeach()

list(PREPEND CMAKE_MODULE_PATH "${SCI_DIR}")
include("${SCI_DIR}/source_of_randomness.cmake")
find_package(OpenSSL REQUIRED)
find_package(GMP MODULE REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
# 覆盖旧缓存，禁止 Eigen 配置从其他安装目录或包注册表回退。
set(Eigen3_DIR "${EIGEN3_DIR}" CACHE PATH "选定的本地 Eigen3 配置目录" FORCE)
find_package(Eigen3 3.3 CONFIG REQUIRED PATHS "${EIGEN3_DIR}" NO_DEFAULT_PATH)
if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "选定 Eigen 配置未导出 Eigen3::Eigen：${EIGEN3_DIR}。")
endif()
include("${SCI_DIR}/SCITargets.cmake")
if(NOT TARGET SCI::SCI-Math)
    message(FATAL_ERROR "SCI 安装未导出 SCI::SCI-Math；请提供兼容的 Math-only build。")
endif()

# Interpreter 和 Development.Module 来自同一 Python，pybind11 也由该解释器提供。
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        RESULT_VARIABLE _pybind11_result
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _pybind11_result EQUAL 0)
        message(FATAL_ERROR "指定 Python 找不到 pybind11；请为 ${Python_EXECUTABLE} 安装 pybind11。")
    endif()
    set(pybind11_DIR "${_pybind11_dir}" CACHE PATH "指定 Python 的 pybind11 CMake 配置目录")
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(litepi_sci litepi_sci.cpp)

# 本地 src/Math/CMakeLists.txt 和 SCITargets.cmake 均导出 Math 及其传递依赖。
target_link_libraries(litepi_sci PRIVATE SCI::SCI-Math)

# 仅安装模块；不自动运行协议或宣称 ARM 硬件已验证。
install(TARGETS litepi_sci
    LIBRARY DESTINATION "${CMAKE_CURRENT_LIST_DIR}/../analysis"
)
