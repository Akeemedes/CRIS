cmake_minimum_required(VERSION 3.20)
project(TwoPhaseTransportTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(twophasetransport_label_oracle src/twophasetransport_label_oracle.cpp)
add_executable(twophasetransport_dataset_packer src/twophasetransport_dataset_packer.cpp)
add_executable(twophasetransport_batch_partitioner src/twophasetransport_batch_partitioner.cpp)

set(TWOPHASETRANSPORT_TORCH_ROOT "" CACHE PATH "Path to the LibTorch distribution used by the native trainer")
if (NOT TWOPHASETRANSPORT_TORCH_ROOT AND DEFINED ENV{LIBTORCH_ROOT})
  set(TWOPHASETRANSPORT_TORCH_ROOT "$ENV{LIBTORCH_ROOT}")
endif()
if (TWOPHASETRANSPORT_TORCH_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${TWOPHASETRANSPORT_TORCH_ROOT}")
  find_package(Torch REQUIRED)
  # The distributed LibTorch headers select AT_PARALLEL_OPENMP. Consumers must
  # also compile with OpenMP or at::parallel_for silently takes its serial
  # fallback, even when the trainer requests multiple threads.
  find_package(OpenMP QUIET)
  add_executable(twophasetransport_lm_trainer src/twophasetransport_lm_trainer.cpp)
  add_executable(twophasetransport_checkpoint_exporter src/twophasetransport_checkpoint_exporter.cpp)
  target_include_directories(twophasetransport_lm_trainer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../native_lm/include")
  target_link_libraries(twophasetransport_lm_trainer PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(twophasetransport_checkpoint_exporter PRIVATE ${TORCH_LIBRARIES})
  target_compile_options(twophasetransport_lm_trainer PRIVATE ${TORCH_CXX_FLAGS})
  target_compile_options(twophasetransport_checkpoint_exporter PRIVATE ${TORCH_CXX_FLAGS})
  # When oneMKL is available, its DSYRK
  # kernel lets the trainer form the symmetric J^T J update without computing
  # the redundant triangle; retain a portable Torch-GEMM fallback otherwise.
  if(TARGET MKL::MKL)
    target_link_libraries(twophasetransport_lm_trainer PRIVATE MKL::MKL)
    target_compile_definitions(twophasetransport_lm_trainer PRIVATE TWOPHASETRANSPORT_HAS_MKL=1)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(twophasetransport_lm_trainer PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(twophasetransport_lm_trainer PRIVATE TWOPHASETRANSPORT_HAS_OPENMP=1)
  else()
    message(WARNING "OpenMP was not found: manual Jacobian and candidate evaluation will use LibTorch's serial fallback")
  endif()
  if(MSVC)
    target_compile_options(twophasetransport_lm_trainer PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(twophasetransport_checkpoint_exporter PRIVATE /W4 /permissive- /EHsc)
  else()
    target_compile_options(twophasetransport_lm_trainer PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(twophasetransport_checkpoint_exporter PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  file(GLOB TWOPHASETRANSPORT_TORCH_DLLS "${TWOPHASETRANSPORT_TORCH_ROOT}/lib/*.dll")
  if(TWOPHASETRANSPORT_TORCH_DLLS)
    add_custom_command(TARGET twophasetransport_lm_trainer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TWOPHASETRANSPORT_TORCH_DLLS}
              "$<TARGET_FILE_DIR:twophasetransport_lm_trainer>")
    add_custom_command(TARGET twophasetransport_checkpoint_exporter POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TWOPHASETRANSPORT_TORCH_DLLS}
              "$<TARGET_FILE_DIR:twophasetransport_checkpoint_exporter>")
  endif()
else()
  message(STATUS "TWOPHASETRANSPORT_TORCH_ROOT not set: building label and packing tools only")
endif()

if(MSVC)
  target_compile_options(twophasetransport_label_oracle PRIVATE /W4 /permissive- /EHsc)
  target_compile_options(twophasetransport_dataset_packer PRIVATE /W4 /permissive- /EHsc)
  target_compile_options(twophasetransport_batch_partitioner PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(twophasetransport_label_oracle PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(twophasetransport_dataset_packer PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(twophasetransport_batch_partitioner PRIVATE -Wall -Wextra -Wpedantic)
endif()
