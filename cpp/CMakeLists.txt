cmake_minimum_required(VERSION 3.18)
project(drone_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 89 = Ada (RTX 4000 Ada), 87 = Jetson Orin. Override with -DCMAKE_CUDA_ARCHITECTURES=...
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
target_link_libraries(trt_bench PRIVATE ${TRT_LIB} CUDA::cudart ${OpenCV_LIBS})
target_compile_options(trt_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
# Find libnvinfer at run time without needing LD_LIBRARY_PATH.
get_filename_component(TRT_LIB_DIR_RESOLVED ${TRT_LIB} DIRECTORY)
set_target_properties(trt_bench PROPERTIES BUILD_RPATH "${TRT_LIB_DIR_RESOLVED}")
