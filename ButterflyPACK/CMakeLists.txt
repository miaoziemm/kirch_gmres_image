cmake_minimum_required(VERSION 3.20)

project(ButterflyPACKCore
    VERSION 4.1.0
    LANGUAGES C CXX Fortran
)

include(GNUInstallDirs)

# Keep the variable names expected by ButterflyPACK's generated precision tree.
set(VERSION_MAJOR "4")
set(VERSION_MINOR "1")
set(VERSION_BugFix "0")
set(ButterflyPACK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ButterflyPACK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

option(enable_mpi "Build ButterflyPACK with MPI/ScaLAPACK" OFF)
option(enable_openmp "Build ButterflyPACK with OpenMP" ON)
option(BPACK_ENABLE_FLOAT "Build complex-float cbutterflypack" ON)
option(BPACK_ENABLE_DOUBLE "Build complex-double zbutterflypack" ON)
option(BPACK_REGENERATE_PRECISIONS
       "Regenerate ButterflyPACK precision sources from SRC" OFF)
option(BPACK_SUPPRESS_WARNINGS
       "Suppress warnings from generated/legacy ButterflyPACK sources" ON)

if(NOT BPACK_ENABLE_FLOAT AND NOT BPACK_ENABLE_DOUBLE)
    message(FATAL_ERROR
        "At least one of BPACK_ENABLE_FLOAT or BPACK_ENABLE_DOUBLE must be ON")
endif()

# Generated precision CMake files expect these compatibility variables.
if(TARGET Kirch::BLAS)
    set(BLAS_LIB Kirch::BLAS)
    set(LAPACK_LIB Kirch::BLAS)
    message(STATUS "ButterflyPACK BLAS/LAPACK target: Kirch::BLAS")
elseif(DEFINED TPL_BLAS_LIBRARIES AND TPL_BLAS_LIBRARIES)
    set(BLAS_LIB ${TPL_BLAS_LIBRARIES})
    if(DEFINED TPL_LAPACK_LIBRARIES AND TPL_LAPACK_LIBRARIES)
        set(LAPACK_LIB ${TPL_LAPACK_LIBRARIES})
    else()
        set(LAPACK_LIB ${TPL_BLAS_LIBRARIES})
    endif()
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_LIB ${BLAS_LIBRARIES})
    set(LAPACK_LIB ${LAPACK_LIBRARIES})
endif()
set(ZFP_LIB "")
set(ARPACK_LIB "")
set(MAGMA_LIB "")
set(SCALAPACK_LIB "")
set(enable_python OFF)

if(enable_mpi)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
    find_package(MPI REQUIRED COMPONENTS C CXX Fortran)

    if(DEFINED TPL_SCALAPACK_LIBRARIES AND TPL_SCALAPACK_LIBRARIES)
        set(SCALAPACK_LIB ${TPL_SCALAPACK_LIBRARIES})
    else()
        find_package(SCALAPACK REQUIRED)
        set(SCALAPACK_LIB ${SCALAPACK_LIBRARIES})
    endif()

    set(MPI_Fortran_LIBRARIES MPI::MPI_Fortran)
    set(MPI_Fortran_INCLUDE_PATH ${MPI_Fortran_INCLUDE_DIRS})
else()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MPI_DUMMY/CMakeLists.txt")
        message(FATAL_ERROR "ButterflyPACK/MPI_DUMMY is required when enable_mpi=OFF")
    endif()
    add_subdirectory(MPI_DUMMY)
endif()

# PrecisionPreprocessing.sh writes compatibility files to EXAMPLE even though
# this trimmed integration does not build the official examples.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/EXAMPLE")

set(_bpack_need_generation ${BPACK_REGENERATE_PRECISIONS})
if(BPACK_ENABLE_FLOAT
   AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SRC_COMPLEX/cBPACK_wrapper.h")
    set(_bpack_need_generation TRUE)
endif()
if(BPACK_ENABLE_DOUBLE
   AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SRC_DOUBLECOMPLEX/zBPACK_wrapper.h")
    set(_bpack_need_generation TRUE)
endif()

if(_bpack_need_generation)
    find_program(BASH_EXECUTABLE bash REQUIRED)
    message(STATUS "Generating ButterflyPACK precision sources")
    execute_process(
        COMMAND "${BASH_EXECUTABLE}" PrecisionPreprocessing.sh OFF
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _bpack_preprocess_result
        OUTPUT_VARIABLE _bpack_preprocess_stdout
        ERROR_VARIABLE _bpack_preprocess_stderr
    )
    if(NOT _bpack_preprocess_result EQUAL 0)
        message(FATAL_ERROR
            "ButterflyPACK precision preprocessing failed.\n"
            "stdout:\n${_bpack_preprocess_stdout}\n"
            "stderr:\n${_bpack_preprocess_stderr}")
    endif()
endif()

set(_bpack_precision_targets "")

if(BPACK_ENABLE_FLOAT)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SRC_COMPLEX/CMakeLists.txt")
        message(FATAL_ERROR "ButterflyPACK/SRC_COMPLEX was not generated")
    endif()
    add_subdirectory(SRC_COMPLEX)
    list(APPEND _bpack_precision_targets cbutterflypack)
    set_target_properties(cbutterflypack PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules-float"
    )
    set_property(TARGET cbutterflypack APPEND PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}/SRC_COMPLEX")
endif()

if(BPACK_ENABLE_DOUBLE)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SRC_DOUBLECOMPLEX/CMakeLists.txt")
        message(FATAL_ERROR "ButterflyPACK/SRC_DOUBLECOMPLEX was not generated")
    endif()
    add_subdirectory(SRC_DOUBLECOMPLEX)
    list(APPEND _bpack_precision_targets zbutterflypack)
    set_target_properties(zbutterflypack PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules-double"
    )
    set_property(TARGET zbutterflypack APPEND PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}/SRC_DOUBLECOMPLEX")
endif()

if(TARGET mpi_dummy)
    set_target_properties(mpi_dummy PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mpi_dummy_modules"
    )
endif()

set(_bpack_compile_targets ${_bpack_precision_targets})
if(TARGET mpi_dummy)
    list(APPEND _bpack_compile_targets mpi_dummy)
endif()

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    foreach(_target IN LISTS _bpack_compile_targets)
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-cpp>
            $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
            $<$<COMPILE_LANGUAGE:Fortran>:-ffixed-line-length-none>
            $<$<COMPILE_LANGUAGE:Fortran>:-fimplicit-none>
            $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>
            $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-fno-range-check>
            $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-fbacktrace>
            $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-fbounds-check>
        )
    endforeach()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel|IntelLLVM")
    foreach(_target IN LISTS _bpack_compile_targets)
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-fpp>
            $<$<COMPILE_LANGUAGE:Fortran>:-extend-source>
        )
    endforeach()
endif()

if(BPACK_SUPPRESS_WARNINGS)
    foreach(_target IN LISTS _bpack_compile_targets)
        if(MSVC)
            target_compile_options(${_target} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:/w>
                $<$<COMPILE_LANGUAGE:CXX>:/w>
            )
        else()
            target_compile_options(${_target} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:-w>
                $<$<COMPILE_LANGUAGE:CXX>:-w>
                $<$<COMPILE_LANGUAGE:Fortran>:-w>
            )
        endif()
    endforeach()
endif()

if(enable_openmp)
    # When ButterflyPACK is built as part of the parent Kirch project, the
    # parent has already discovered or created the OpenMP imported targets.
    # This is especially important on macOS: AppleClang needs Homebrew libomp,
    # and rerunning FindOpenMP here discards that fallback and fails with
    # OpenMP_CXX_LIB_NAMES missing.  Only discover components that the parent
    # has not provided (this also keeps standalone ButterflyPACK builds valid).
    set(_bpack_missing_openmp_components)
    if(NOT TARGET OpenMP::OpenMP_CXX)
        list(APPEND _bpack_missing_openmp_components CXX)
    endif()
    if(NOT TARGET OpenMP::OpenMP_Fortran)
        list(APPEND _bpack_missing_openmp_components Fortran)
    endif()

    if(_bpack_missing_openmp_components)
        find_package(OpenMP REQUIRED
            COMPONENTS ${_bpack_missing_openmp_components})
    else()
        message(STATUS
            "ButterflyPACK reusing parent OpenMP CXX and Fortran targets")
    endif()

    if(NOT TARGET OpenMP::OpenMP_CXX OR
       NOT TARGET OpenMP::OpenMP_Fortran)
        message(FATAL_ERROR
            "ButterflyPACK requires OpenMP::OpenMP_CXX and OpenMP::OpenMP_Fortran")
    endif()

    foreach(_target IN LISTS _bpack_precision_targets)
        target_compile_definitions(${_target} PRIVATE HAVE_OPENMP)
        # Generated precision CMake files use the plain link signature.
        target_link_libraries(${_target}
            OpenMP::OpenMP_CXX
            OpenMP::OpenMP_Fortran
        )
    endforeach()

    unset(_bpack_missing_openmp_components)
endif()

if(TARGET cbutterflypack)
    add_library(ButterflyPACK::cbutterflypack ALIAS cbutterflypack)
    add_library(ButterflyPACK::float ALIAS cbutterflypack)
endif()
if(TARGET zbutterflypack)
    add_library(ButterflyPACK::zbutterflypack ALIAS zbutterflypack)
    add_library(ButterflyPACK::double ALIAS zbutterflypack)
endif()
