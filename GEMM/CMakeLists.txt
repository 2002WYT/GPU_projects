cmake_minimum_required(VERSION 3.18)
if(DEFINED ENV{CUDA_HOME})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA Toolkit root")
endif()
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CUDSS_ROOT /home/wangyitong/cudss_install/nvidia/cu12)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${CUDSS_ROOT}/include
)

link_directories(
    ${catkin_LIB_DIRS}
    ${CUDSS_ROOT}/lib
)

find_package(CUDAToolkit REQUIRED)
aux_source_directory(src SRC_SOURCE)
add_library(SRC ${SRC_SOURCE})


set(EXE2 gemm_autotune)
add_executable(${EXE2} gemm_compare_autotune.cu)

target_link_libraries(${EXE2} PRIVATE
    SRC
    CUDA::cublas
    CUDA::cudart
)

set(EXE3 gemm_selected)
add_executable(${EXE3} gemm_selected_config.cu)

target_link_libraries(${EXE3} PRIVATE
    SRC
    CUDA::cublas
    CUDA::cudart
)

set(EXE4 gemm_mtx)
add_executable(${EXE4} gemm_mtx_operator.cu)

target_link_libraries(${EXE4} PRIVATE
    SRC
    CUDA::cublas
    CUDA::cudart
)

foreach(target ${EXE2} ${EXE3} ${EXE4})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endforeach()
