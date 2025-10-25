#
# CMakeLists.txt - CMake configuration file for swmm-solver
#
# Created: July 11, 2019
# Modified: Aug 16, 2022
#
# Author: Michael E. Tryby
#         US EPA ORD/CESER
#


cmake_minimum_required (VERSION 3.13)

if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR "In-source builds are disabled.")
endif()


# Check for GPU support option before project definition
option(BUILD_GPU "Enable GPU acceleration with CUDA" OFF)

# Determine languages based on BUILD_GPU option
if(BUILD_GPU)
    project(swmm-solver
        VERSION 5.2.4
        LANGUAGES C CXX CUDA
    )
    message(STATUS "SWMM-GPU: CUDA support enabled")
else()
    project(swmm-solver
        VERSION 5.2.4
        LANGUAGES C CXX
    )
endif()

# Append local dir to module search path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Sets the position independent code property for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Sets default install prefix when cmakecache is initialized for first time
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()


# Define install locations (will be prepended by install prefix)
set(TOOL_DIST "bin")
set(INCLUDE_DIST "include")
set(LIBRARY_DIST "lib")
set(CONFIG_DIST  "cmake")


# Define build options
option(BUILD_TESTS "Builds component tests (requires Boost)" OFF)
option(BUILD_DEF   "Builds library with def file interface" OFF)

# CUDA/GPU configuration
if(BUILD_GPU)
    # Set CUDA standard and architectures
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Support multiple GPUs: 89 (RTX 4060 laptop), 121 (GB10 DGX Spark)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "89;121" CACHE STRING "CUDA architectures to build for")
    endif()

    # Enable separable compilation for device code
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Add GPU compile definition
    add_compile_definitions(BUILD_GPU)

    message(STATUS "SWMM-GPU: Target architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()


# Add project subdirectories
add_subdirectory(src/outfile)
add_subdirectory(src/solver)
add_subdirectory(src/run)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


# Create install rules for vcruntime.dll, msvcp.dll, vcomp.dll etc.
set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
include(InstallRequiredSystemLibraries)


# Configure CPack driven installer package
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VENDOR "US_EPA")
set(CPACK_ARCHIVE_FILE_NAME "swmm")

include(CPack)
