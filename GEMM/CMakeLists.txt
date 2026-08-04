cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(GPUGEMM LANGUAGES CXX CUDA)
    include(CTest)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
            "CUDA architectures to build for (70 targets NVIDIA V100)")
    endif()
endif()

foreach(target IN ITEMS gemm_autotune gemm_selected gemm_mtx)
    if(target STREQUAL "gemm_autotune")
        set(source gemm_compare_autotune.cu)
    elseif(target STREQUAL "gemm_selected")
        set(source gemm_selected_config.cu)
    else()
        set(source gemm_mtx_operator.cu)
    endif()

    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::cudart)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()
