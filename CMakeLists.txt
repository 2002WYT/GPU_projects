cmake_minimum_required(VERSION 3.24)

project(GPUProjects VERSION 1.0.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
        "CUDA architectures to build for (70 targets NVIDIA V100)")
endif()

option(GPU_PROJECTS_BUILD_SPMV "Build the SpMV benchmark" ON)
option(GPU_PROJECTS_BUILD_JACOBI "Build the Jacobi solver" ON)
option(GPU_PROJECTS_BUILD_GAUSS_SEIDEL "Build the multicolor Gauss-Seidel solver" ON)
option(GPU_PROJECTS_BUILD_CG "Build the Conjugate Gradient solver" ON)
option(GPU_PROJECTS_BUILD_GEMM "Build the GEMM benchmarks" ON)
option(GPU_PROJECTS_BUILD_SOLVERS "Build the optional sparse direct-solver adapters" OFF)

function(gpu_projects_configure_target target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
endfunction()

if(GPU_PROJECTS_BUILD_SPMV)
    add_executable(spmv_compare spmv/spmv_compare_complete.cu)
    gpu_projects_configure_target(spmv_compare)

    if(BUILD_TESTING)
        add_test(NAME spmv_smoke
            COMMAND spmv_compare 4096 2 1 0 128)
    endif()
endif()

if(GPU_PROJECTS_BUILD_JACOBI)
    add_executable(jacobi_solver
        jacobi/ex_jacobi.cu
        jacobi/src/jacobi.cu)
    target_include_directories(jacobi_solver PRIVATE jacobi/include)
    set_target_properties(jacobi_solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    gpu_projects_configure_target(jacobi_solver)

    if(BUILD_TESTING)
        add_test(NAME jacobi_smoke COMMAND jacobi_solver 4096)
    endif()
endif()

if(GPU_PROJECTS_BUILD_GAUSS_SEIDEL)
    add_executable(gauss_seidel_solver
        gauss-seidel/ex_gs.cu
        gauss-seidel/src/gs.cu)
    target_include_directories(gauss_seidel_solver PRIVATE gauss-seidel/include)
    gpu_projects_configure_target(gauss_seidel_solver)

    if(BUILD_TESTING)
        add_test(NAME gauss_seidel_smoke COMMAND gauss_seidel_solver 4096)
    endif()
endif()

if(GPU_PROJECTS_BUILD_CG)
    add_executable(cg_solver
        CG/ex_CG.cu
        CG/src/CG.cu
        CG/src/math.cu)
    target_include_directories(cg_solver PRIVATE CG/include)
    set_target_properties(cg_solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    gpu_projects_configure_target(cg_solver)

    if(BUILD_TESTING)
        add_test(NAME cg_smoke COMMAND cg_solver 4096)
    endif()
endif()

if(GPU_PROJECTS_BUILD_GEMM)
    add_subdirectory(GEMM)
endif()

if(GPU_PROJECTS_BUILD_SOLVERS)
    add_subdirectory("Four sparse solvers")
endif()
