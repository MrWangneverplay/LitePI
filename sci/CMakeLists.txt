cmake_minimum_required(VERSION 3.15)
project(litepi_sci LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default: <repo>/third_party/EzPC/SCI/install (scripts/build_sci.sh installs here).
if(NOT SCI_INSTALL_DIR)
  get_filename_component(SCI_INSTALL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/EzPC/SCI/install" ABSOLUTE)
endif()
if(NOT EXISTS "${SCI_INSTALL_DIR}/include/Math/math-functions.h")
  message(FATAL_ERROR "SCI headers not found under ${SCI_INSTALL_DIR}; run scripts/build_sci.sh")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(litepi_sci litepi_sci.cpp)

# Drop the extension next to sci_relu.py, which adds <repo>/sci to sys.path.
set_target_properties(litepi_sci PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(litepi_sci PRIVATE "${SCI_INSTALL_DIR}/include")
target_link_directories(litepi_sci PRIVATE "${SCI_INSTALL_DIR}/lib")

# SCI's headers rely on these ISA flags (see SCI src/utils/CMakeLists.txt) and
# on rdseed being available; CMake falls back to random_device otherwise.
target_compile_options(litepi_sci PRIVATE
  -O3 -maes -msse4.1 -mavx -mavx2 -mrdseed -fopenmp
  -Wno-deprecated-declarations)

target_link_libraries(litepi_sci PRIVATE
  SCI-OT
  SCI-Math
  SCI-BuildingBlocks
  SCI-LinearOT
  SCI-FloatingPoint
  SCI-GC
  # SCI-GC and SCI-OT reference each other; repeating the OT archive after GC
  # resolves the cycle without needing --start-group.
  ${SCI_INSTALL_DIR}/lib/libSCI-OT.a
  gmp
  ssl
  crypto
  OpenMP::OpenMP_CXX
  Threads::Threads)
