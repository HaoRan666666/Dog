cmake_minimum_required(VERSION 3.15)
project(raycaster_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# 固定用 mj conda 环境自带的 mujoco pip 包里的 C SDK，
# 必须和 Python 里 import mujoco 用的是同一份 libmujoco.so，
# 否则 mjModel/mjData 的 struct 布局可能不一致（ABI 不匹配）。
set(MJ_PY_PKG_DIR "$ENV{HOME}/miniconda3/envs/mj/lib/python3.12/site-packages/mujoco"
    CACHE PATH "mj conda env 里 mujoco pip 包的目录")

if(NOT EXISTS "${MJ_PY_PKG_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "找不到 ${MJ_PY_PKG_DIR}/include/mujoco/mujoco.h，"
    "请用 -DMJ_PY_PKG_DIR=... 指定正确的 mujoco pip 包目录")
endif()

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

set(RAYCASTER_SRC_DIR "/home/rp/开源/code/mujoco_ray_caster/raycaster_src")
file(GLOB RAYCASTER_SOURCES
  "${RAYCASTER_SRC_DIR}/RayCaster.cpp"
  "${RAYCASTER_SRC_DIR}/RayCasterCamera.cpp"
)

pybind11_add_module(raycaster_ext
  binding.cpp
  ${RAYCASTER_SOURCES}
)

target_include_directories(raycaster_ext PRIVATE
  "${RAYCASTER_SRC_DIR}"
  "${MJ_PY_PKG_DIR}/include"
)

target_link_libraries(raycaster_ext PRIVATE
  "${MJ_PY_PKG_DIR}/libmujoco.so.3.5.0"
)

# 运行时按 SONAME (libmujoco.so.3.5.0) 去找库，加 rpath 指到同一份 so，
# 避免加载到系统里其它版本的 libmujoco。
set_target_properties(raycaster_ext PROPERTIES
  INSTALL_RPATH "${MJ_PY_PKG_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# 编译产物直接落到 sim2sim/ 目录，方便 python 直接 import。
set_target_properties(raycaster_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
