cmake_minimum_required(VERSION 3.20)

# The in-tree bf1d phase-aware factorization is C++; the optional external
# ButterflyPACK backend alone requires Fortran.  Allow lightweight builds on
# systems that intentionally disable that backend.
option(KIRCH_ENABLE_BUTTERFLYPACK
       "Build ButterflyPACK complex single/double precision cores" ON)
set(_kirch_project_languages C CXX)
if(KIRCH_ENABLE_BUTTERFLYPACK)
    list(APPEND _kirch_project_languages Fortran)
endif()

project(KirchRTMBFMigration
    VERSION 1.0.0
    DESCRIPTION "Kirchhoff RTM and ButterflyPACK acceleration"
    LANGUAGES ${_kirch_project_languages}
)

include(GNUInstallDirs)
include(CTest)

# -----------------------------------------------------------------------------
# Language and output layout
# -----------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Project options
# -----------------------------------------------------------------------------
option(KIRCH_ENABLE_OPENMP "Enable OpenMP" ON)
option(KIRCH_ENABLE_MPI "Enable MPI and ScaLAPACK ButterflyPACK path" OFF)
option(KIRCH_BPACK_ENABLE_FLOAT
       "Build ButterflyPACK complex-float target cbutterflypack" ON)
option(KIRCH_BPACK_ENABLE_DOUBLE
       "Build ButterflyPACK complex-double target zbutterflypack" ON)
option(KIRCH_REGENERATE_BUTTERFLYPACK
       "Regenerate ButterflyPACK precision sources from ButterflyPACK/SRC" OFF)
option(KIRCH_BUILD_APPLICATIONS "Build application subdirectories when present" ON)
option(KIRCH_ENABLE_INSTALL "Enable install rules" OFF)
option(KIRCH_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
option(KIRCH_SUPPRESS_THIRD_PARTY_WARNINGS
       "Suppress compiler warnings emitted by bundled third-party sources" ON)

option(KIRCH_ENABLE_NATIVE_OPTIMIZATION
       "Enable host-specific CPU tuning for project and ButterflyPACK targets" OFF)

set(KIRCH_EIGEN_PROVIDER "BUNDLED" CACHE STRING
    "Eigen provider: BUNDLED or SYSTEM")
set_property(CACHE KIRCH_EIGEN_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM)

set(KIRCH_SUPERLU_PROVIDER "BUNDLED" CACHE STRING
    "SuperLU provider used by the GMRES preconditioner: BUNDLED or SYSTEM")
set_property(CACHE KIRCH_SUPERLU_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM)

set(KIRCH_FFTW_PROVIDER "BUNDLED" CACHE STRING
    "FFTW provider: BUNDLED or SYSTEM")
set_property(CACHE KIRCH_FFTW_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM)

if(APPLE)
    set(_kirch_default_blas_provider "SYSTEM")
else()
    set(_kirch_default_blas_provider "BUNDLED")
endif()
set(KIRCH_BLAS_PROVIDER "${_kirch_default_blas_provider}" CACHE STRING
    "BLAS/LAPACK provider used by ButterflyPACK: BUNDLED or SYSTEM")
set_property(CACHE KIRCH_BLAS_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM)
unset(_kirch_default_blas_provider)

option(KIRCH_ALLOW_BUNDLED_OPENBLAS_ON_APPLE
       "Allow the bundled OpenBLAS build on macOS instead of Accelerate" OFF)

option(KIRCH_OPENBLAS_DYNAMIC_ARCH
       "Build OpenBLAS with runtime CPU dispatch for multiple architectures" OFF)
option(KIRCH_OPENBLAS_USE_THREADS
       "Enable OpenBLAS internal pthread parallelism" OFF)

if(KIRCH_ENABLE_BUTTERFLYPACK
   AND NOT KIRCH_BPACK_ENABLE_FLOAT
   AND NOT KIRCH_BPACK_ENABLE_DOUBLE)
    message(FATAL_ERROR
        "KIRCH_ENABLE_BUTTERFLYPACK=ON requires at least one of "
        "KIRCH_BPACK_ENABLE_FLOAT or KIRCH_BPACK_ENABLE_DOUBLE")
endif()

# The retained ButterflyPACK 4.1 sources call OpenMP timing/thread routines in
# several translation units. Keep OpenMP enabled for this integration; a
# one-thread serial run can still be requested with OMP_NUM_THREADS=1.
if(KIRCH_ENABLE_BUTTERFLYPACK AND NOT KIRCH_ENABLE_OPENMP)
    message(FATAL_ERROR
        "The bundled ButterflyPACK integration requires KIRCH_ENABLE_OPENMP=ON. "
        "Use OMP_NUM_THREADS=1 for serial execution.")
endif()

# Static third-party libraries are simpler for this research codebase.
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(KirchDependencies)
include(KirchCompilerOptions)
include(KirchEigen)

include(KirchSuperLU)
# -----------------------------------------------------------------------------
# Core libraries
# -----------------------------------------------------------------------------
add_subdirectory(SELINEAR)
add_subdirectory(SEBASIC)
add_subdirectory(SEFILESYSTEM)
add_subdirectory(SERECKIRCH)

# Add application directories only when they are present.
if(KIRCH_BUILD_APPLICATIONS)
    foreach(_app_dir IN ITEMS apps src examples)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_app_dir}/CMakeLists.txt")
            add_subdirectory("${_app_dir}")
        endif()
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Optional installation
# -----------------------------------------------------------------------------
if(KIRCH_ENABLE_INSTALL)
    install(TARGETS
        selinear
        sebasic
        sefs
        reckirch
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )

    if(KIRCH_EIGEN_PROVIDER STREQUAL "BUNDLED")
        install(DIRECTORY
            "${PROJECT_SOURCE_DIR}/eigen/Eigen"
            "${PROJECT_SOURCE_DIR}/eigen/unsupported"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/eigen3")
    endif()
endif()

message(STATUS "")
message(STATUS "========== KirchRTMBFMigration configuration ==========")
message(STATUS "Build type             : ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler             : ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler           : ${CMAKE_CXX_COMPILER}")
message(STATUS "Fortran compiler       : ${CMAKE_Fortran_COMPILER}")
message(STATUS "OpenMP                 : ${KIRCH_ENABLE_OPENMP}")
message(STATUS "MPI                    : ${KIRCH_ENABLE_MPI}")
message(STATUS "Eigen provider         : ${KIRCH_EIGEN_PROVIDER}")
message(STATUS "Eigen version          : ${KIRCH_EIGEN_VERSION}")
message(STATUS "FFTW provider          : ${KIRCH_FFTW_PROVIDER}")
message(STATUS "FFTW precisions        : float + double")
message(STATUS "BLAS provider          : ${KIRCH_BLAS_PROVIDER}")
message(STATUS "SuperLU provider       : ${KIRCH_SUPERLU_PROVIDER}")
message(STATUS "OpenBLAS precisions    : S + D + C + Z")
message(STATUS "OpenBLAS dynamic arch  : ${KIRCH_OPENBLAS_DYNAMIC_ARCH}")
message(STATUS "OpenBLAS threads       : ${KIRCH_OPENBLAS_USE_THREADS}")
message(STATUS "ButterflyPACK          : ${KIRCH_ENABLE_BUTTERFLYPACK}")
message(STATUS "BPACK float            : ${KIRCH_BPACK_ENABLE_FLOAT}")
message(STATUS "BPACK double           : ${KIRCH_BPACK_ENABLE_DOUBLE}")
message(STATUS "Suppress 3rd warnings  : ${KIRCH_SUPPRESS_THIRD_PARTY_WARNINGS}")
message(STATUS "Native CPU tuning      : ${KIRCH_ENABLE_NATIVE_OPTIMIZATION}")
message(STATUS "Build applications     : ${KIRCH_BUILD_APPLICATIONS}")
message(STATUS "=======================================================")
