# =============================================================================
# Basic Configuration
# =============================================================================

# Minimum required CMake version
cmake_minimum_required(VERSION 3.18)

# Set policy to avoid compiler ID conflicts
cmake_policy(SET CMP0069 NEW)

# Project settings
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")

# HIP/ROCm support option
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
    # The HIP language needs 3.21; only this branch does, so CUDA users on 3.18-3.20 are
    # unaffected. Pass -DCMAKE_HIP_ARCHITECTURES=gfx90a (etc.) to override the default,
    # which CMake takes from the GPUs present on the build host.
    cmake_minimum_required(VERSION 3.21)

    # Initialize project with C++ and HIP languages
    project(gpu_ipc LANGUAGES CXX HIP)
else()
    set(CUDA_ARCHITECTURES 86)  # SM86 for Ampere architecture (RTX 30xx series)

    # Initialize project with C++ and CUDA languages
    project(gpu_ipc LANGUAGES CXX CUDA)
endif()

# =============================================================================
# Language Standards Configuration
# =============================================================================

# Set C++17 standard with strict compliance
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_HIP)
    # Set HIP C++17 standard
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
else()
    # Set CUDA 17 standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# =============================================================================
# Dependency Management
# =============================================================================

if(USE_HIP)
    # hip::host carries the ROCm include directories and __HIP_PLATFORM_AMD__, so the
    # plain C++ sources see the same runtime declarations as the HIP sources without
    # anything naming an install prefix.
    find_package(hip REQUIRED)
else()
    # Find CUDA Toolkit with required components (not needed for HIP)
    find_package(CUDAToolkit REQUIRED)
endif()

# Find third-party dependencies
find_package(Eigen3 REQUIRED)    # Linear algebra library
find_package(GLEW REQUIRED)      # OpenGL Extension Wrangler
find_package(GLUT REQUIRED)      # OpenGL Utility Toolkit
find_package(OpenGL REQUIRED)    # Open Graphics Library

# =============================================================================
# Target Configuration
# =============================================================================

# Create main executable target
add_executable(gipc)

if(USE_HIP)
    # Set HIP-specific target properties
    # Enable relocatable device code for cross-TU device symbol linking
    set_target_properties(gipc PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_SEPARABLE_COMPILATION ON
    )
else()
    # Set CUDA-specific target properties
    set_target_properties(gipc PROPERTIES
        CUDA_EXTENSIONS OFF                    # Disable CUDA compiler extensions
        CUDA_SEPARABLE_COMPILATION ON          # Enable separate compilation
        CUDA_RESOLVE_DEVICE_SYMBOLS ON         # Resolve device symbols at link time
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} # Target CUDA architecture
    )
endif()

# =============================================================================
# Source File Management
# =============================================================================

# Recursively collect all CUDA and C++ source files from GPU_IPC directory
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS
    "GPU_IPC/*.cu"     # CUDA source files
    "GPU_IPC/*.cpp"    # C++ source files
)

# Recursively collect all header files from GPU_IPC directory
file(GLOB_RECURSE GIPC_HEADER CONFIGURE_DEPENDS
    "GPU_IPC/*.h"      # C header files
    "GPU_IPC/*.cuh"    # CUDA header files
    "GPU_IPC/*.hpp"    # C++ header files
    "GPU_IPC/*.inl"    # Inline files
    "GPU_IPC/*.inc"    # Include files
)

# For HIP builds, mark .cu files as HIP language
if(USE_HIP)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "GPU_IPC/*.cu")
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

# Add all collected sources to the target
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HEADER})

# =============================================================================
# Include Directories
# =============================================================================

# Set include directories for the target
target_include_directories(gipc PRIVATE
    "GPU_IPC/"                 # Main source directory
    ${Eigen3_INCLUDE_DIRS}     # Eigen3 include directories
)

# =============================================================================
# Compile Definitions
# =============================================================================

# Set preprocessor definitions
target_compile_definitions(gipc PRIVATE
    GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"    # Assets directory path
    GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/"    # Output directory path
    USE_SNK                   # Enable SNK feature
    ADAPTIVE_KAPPA           # Enable adaptive kappa
    $<$<BOOL:${USE_HIP}>:USE_HIP>  # Define USE_HIP when building for AMD
)

# =============================================================================
# Library Linking
# =============================================================================

if(USE_HIP)
    target_link_libraries(gipc PRIVATE hip::host)
else()
    # Link CUDA libraries (only cudart is actually used; others are dead code)
    target_link_libraries(gipc PRIVATE
        CUDA::cudart      # CUDA runtime library
        CUDA::cublas      # CUDA BLAS library (unused but kept for CUDA build)
        CUDA::cusolver    # CUDA solver library (unused but kept for CUDA build)
        CUDA::cusparse    # CUDA sparse matrix library (unused but kept for CUDA build)
        CUDA::cufft       # CUDA FFT library (unused but kept for CUDA build)
    )
endif()
# HIP runtime is linked automatically by the HIP language support

# Link graphics and math libraries
target_link_libraries(gipc PRIVATE
    GLUT::GLUT        # GLUT library
    GLEW::GLEW        # GLEW library
    OpenGL::GLU    # OpenGL library
    Eigen3::Eigen     # Eigen3 library
)

# =============================================================================
# Compiler Options
# =============================================================================

if(USE_HIP)
    # clang's -ffast-math is far broader than nvcc's --use_fast_math: it also implies
    # finite-math-only, reassociation and reciprocal math for double precision, which the
    # continuous-collision and intersection predicates here cannot tolerate. -ffp-contract
    # is pinned to expression scope for the same reason: clang defaults it to "fast" when
    # compiling HIP and then contracts across expressions, which nvcc never does.
    target_compile_options(gipc PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            -ffp-contract=on         # Contract within one expression, matching nvcc
            -fgpu-rdc                # Enable relocatable device code
        >
    )
    # Linker flags for device code linking
    target_link_options(gipc PRIVATE -fgpu-rdc)
else()
    # Set CUDA-specific compiler options
    target_compile_options(gipc PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:  # Apply only to CUDA files
            --use_fast_math          # Use fast math operations
            -lineinfo                # Generate line information for profiling
            --ptxas-options=-allow-expensive-optimizations=true,-v  # PTX assembler options
            --extended-lambda        # Enable extended lambda support
            --expt-relaxed-constexpr # Enable relaxed constexpr support
            --default-stream=per-thread  # Use per-thread default stream
        >
    )
endif()

# =============================================================================
# IDE Support and Organization
# =============================================================================

# Organize source files in IDE projects
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_HEADER})

# =============================================================================
# Configuration Summary
# =============================================================================

# Display configuration information
message(STATUS "Project: gpu_ipc")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(USE_HIP)
    message(STATUS "Backend: HIP/ROCm")
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
    message(STATUS "HIP standard: ${CMAKE_HIP_STANDARD}")
else()
    message(STATUS "Backend: CUDA")
    message(STATUS "CUDA architectures: ${CUDA_ARCHITECTURES}")
    message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Source files found: ${GIPC_SOURCE}")
message(STATUS "Header files found: ${GIPC_HEADER}")