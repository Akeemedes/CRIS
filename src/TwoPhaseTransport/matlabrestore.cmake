cmake_minimum_required(VERSION 3.18)
project(TwoPhaseTransport LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------- libtorch (Windows) --------------------
set(TORCH_PATH "${CMAKE_CURRENT_LIST_DIR}/../libtorch")
get_filename_component(TORCH_PATH "${TORCH_PATH}" ABSOLUTE)
if (NOT EXISTS "${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
  message(FATAL_ERROR "TorchConfig.cmake not found under: ${TORCH_PATH}\n"
                      "Expected: ${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PATH}")
find_package(Torch REQUIRED)

# -------------------- target --------------------
add_executable(tp_transport main.cpp)

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

target_compile_options(tp_transport PRIVATE /EHsc)

# -------------------- copy Torch DLLs next to exe --------------------
file(GLOB TORCH_DLLS "${TORCH_PATH}/lib/*.dll")
if (TORCH_DLLS)
  add_custom_command(TARGET tp_transport POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            "$<TARGET_FILE_DIR:tp_transport>"
  )
endif()
