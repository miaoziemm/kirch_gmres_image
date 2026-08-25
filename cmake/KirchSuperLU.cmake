# The retained imaging path uses SuperLU ILUTP for its global shifted-
# Laplacian preconditioner.  Build only the library, not upstream examples or
# tests, and keep the bundled project independent of a Fortran compiler.
if(NOT KIRCH_ENABLE_BUTTERFLYPACK)
    return()
endif()

set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_tests OFF CACHE BOOL "" FORCE)
set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_internal_blaslib OFF CACHE BOOL "" FORCE)
set(SUPERLU_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
if(NOT TARGET Kirch::BLAS)
    message(FATAL_ERROR "Global imaging requires the Kirch::BLAS target")
endif()
# SuperLU checks its TPL_BLAS_LIBRARIES cache entry rather than an existing
# CMake target.  Supplying the target here prevents a redundant system-BLAS
# search and prevents SuperLU from silently building its small internal BLAS.
set(TPL_BLAS_LIBRARIES Kirch::BLAS CACHE STRING "" FORCE)
set(BLAS_FOUND TRUE)
set(METIS_LIB "")
add_subdirectory("${PROJECT_SOURCE_DIR}/superlu"
                 "${PROJECT_BINARY_DIR}/_deps/SuperLU")

if(TARGET superlu)
    add_library(Kirch::SuperLU ALIAS superlu)
endif()
