cmake_minimum_required(VERSION 3.18)
project(drone_trt LANGUAGES CXX)

# DRONE_TRT_BUILD_RUNTIME: the CUDA/TensorRT benchmark (trt_bench); needs CUDA, TensorRT and
#   OpenCV. ON by default, so the documented GPU build is unchanged.
# DRONE_TRT_BUILD_TESTS: CPU-only GoogleTest unit tests for decode/NMS, letterbox geometry and
#   engine-file parsing. Configure with -DDRONE_TRT_BUILD_RUNTIME=OFF -DDRONE_TRT_BUILD_TESTS=ON
#   on a machine without CUDA.
option(DRONE_TRT_BUILD_RUNTIME "Build the CUDA/TensorRT benchmark (needs CUDA, TensorRT, OpenCV)" ON)
option(DRONE_TRT_BUILD_TESTS "Build the CPU unit tests (GoogleTest)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Plain C++ (no CUDA or TensorRT headers): decode + NMS, letterbox geometry, engine-file parsing.
add_library(drone_trt_core STATIC
  src/letterbox.cpp
  src/postprocess.cpp
)
target_include_directories(drone_trt_core PUBLIC src)
target_compile_options(drone_trt_core PRIVATE -O3 -Wall -Wextra)

if(DRONE_TRT_BUILD_RUNTIME)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # 89 = Ada (RTX 4000 Ada), 87 = Jetson Orin. Override with -DCMAKE_CUDA_ARCHITECTURES=...
  # Note: enable_language(CUDA) already initialises CMAKE_CUDA_ARCHITECTURES to nvcc's default
  # (CMake policy CMP0104), so this fallback only applies if that did not happen. Kept as in
  # the build that produced the published results; to be made explicit when they are re-measured.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()

  find_package(CUDAToolkit REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

  # TensorRT: point these at the headers and libraries of the SAME TensorRT version
  # that built the .engine files (engines are not portable across TensorRT versions).
  set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory containing NvInfer.h")
  set(TENSORRT_LIB_DIR "" CACHE PATH "Directory containing libnvinfer.so*")
  find_path(TRT_INCLUDE NvInfer.h HINTS ${TENSORRT_INCLUDE_DIR} REQUIRED)
  find_library(TRT_LIB NAMES nvinfer libnvinfer.so.11 libnvinfer.so.10 HINTS ${TENSORRT_LIB_DIR} REQUIRED)
  message(STATUS "TensorRT headers: ${TRT_INCLUDE}")
  message(STATUS "TensorRT library: ${TRT_LIB}")

  add_executable(trt_bench
    src/main.cpp
    src/detector.cpp
    src/preprocess.cu
  )
  target_include_directories(trt_bench PRIVATE src)
  # SYSTEM: silence warnings coming from third-party headers
  target_include_directories(trt_bench SYSTEM PRIVATE ${TRT_INCLUDE} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(trt_bench PRIVATE drone_trt_core ${TRT_LIB} CUDA::cudart ${OpenCV_LIBS})
  target_compile_options(trt_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
  )
  # Find libnvinfer at run time without needing LD_LIBRARY_PATH.
  get_filename_component(TRT_LIB_DIR_RESOLVED ${TRT_LIB} DIRECTORY)
  set_target_properties(trt_bench PROPERTIES BUILD_RPATH "${TRT_LIB_DIR_RESOLVED}")
endif()

if(DRONE_TRT_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(core_tests
    tests/test_postprocess.cpp
    tests/test_letterbox.cpp
  )
  target_link_libraries(core_tests PRIVATE drone_trt_core GTest::gtest_main)
  target_compile_options(core_tests PRIVATE -Wall -Wextra)
  include(GoogleTest)
  gtest_discover_tests(core_tests)
endif()
