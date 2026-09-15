cmake_minimum_required(VERSION 3.18)
project(TwoPhaseTransport LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------- LibTorch --------------------
# Set CRIS_TORCH_ROOT at configure time, or set the LIBTORCH_ROOT environment
# variable.  Keeping the dependency outside the repository avoids committing a
# large platform-specific binary distribution.
set(CRIS_TORCH_ROOT "" CACHE PATH "Path to the LibTorch distribution root")
if (NOT CRIS_TORCH_ROOT AND DEFINED ENV{LIBTORCH_ROOT})
  set(CRIS_TORCH_ROOT "$ENV{LIBTORCH_ROOT}")
endif()
if (NOT CRIS_TORCH_ROOT)
  message(FATAL_ERROR
    "LibTorch is required. Configure with -DCRIS_TORCH_ROOT=<libtorch-root>, "
    "or set the LIBTORCH_ROOT environment variable.")
endif()
get_filename_component(TORCH_PATH "${CRIS_TORCH_ROOT}" ABSOLUTE)
if (NOT EXISTS "${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
  message(FATAL_ERROR "TorchConfig.cmake not found under: ${TORCH_PATH}\n"
                      "Expected: ${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PATH}")
find_package(Torch REQUIRED)

# -------------------- target --------------------
add_executable(tp_transport main.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../CRIS/ExactRegularLocalInverse.cpp")

get_filename_component(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${SRC_ROOT}/CORE_SCL/*.hpp" "${SRC_ROOT}/CORE_SCL/*.h"
  "${SRC_ROOT}/CRIS/*.hpp"     "${SRC_ROOT}/CRIS/*.h"
)

target_sources(tp_transport PRIVATE ${PROJECT_HEADERS})
source_group(TREE "${SRC_ROOT}" FILES ${PROJECT_HEADERS})

target_include_directories(tp_transport PRIVATE
  "${SRC_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${SRC_ROOT}/CORE_SCL"
  "${SRC_ROOT}/CRIS"
)

target_link_libraries(tp_transport PRIVATE ${TORCH_LIBRARIES})

if (MSVC)
  target_compile_options(tp_transport PRIVATE /EHsc)
  # The Torch-heavy CRIS headers and the exact-reference path exhaust MSVC's
  # default compiler heap in a single translation unit on Windows.
  target_compile_options(tp_transport PRIVATE /Zm200)
endif()

# -------------------- copy Torch DLLs next to exe --------------------
file(GLOB TORCH_DLLS "${TORCH_PATH}/lib/*.dll")
if (TORCH_DLLS)
  add_custom_command(TARGET tp_transport POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            "$<TARGET_FILE_DIR:tp_transport>"
  )
endif()
