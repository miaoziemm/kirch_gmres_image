find_package(Threads REQUIRED)

# Build both precisions required by the wave and Kirchhoff implementations.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_THREADS OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(ENABLE_FLOAT OFF CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/fftw" "${PROJECT_BINARY_DIR}/_deps/fftw-double")
set(KIRCH_FFTW_DOUBLE_TARGET fftw3)
# FFTW's portable cycle counter uses the GNU/Clang `asm` spelling on AArch64
# (including Apple Silicon).  The main project intentionally uses strict C11,
# but this bundled third-party target must retain compiler C extensions.
set_target_properties(fftw3 PROPERTIES C_EXTENSIONS ON)

set(ENABLE_FLOAT ON CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/fftw" "${PROJECT_BINARY_DIR}/_deps/fftw-float")
set(KIRCH_FFTW_FLOAT_TARGET fftw3f)
set_target_properties(fftw3f PROPERTIES C_EXTENSIONS ON)

if(KIRCH_ENABLE_BUTTERFLYPACK)
    if(KIRCH_BLAS_PROVIDER STREQUAL "BUNDLED")
        # ButterflyPACK requires both BLAS and LAPACK.  Configure the bundled
        # OpenBLAS before ButterflyPACK so a machine without a system BLAS can
        # configure the default project successfully.
        set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
        set(BUILD_WITHOUT_LAPACKE ON CACHE BOOL "" FORCE)
        # This source snapshot intentionally omits LAPACK's TESTING/MATGEN
        # directory.  Those matrix generators are not part of the runtime
        # BLAS/LAPACK API required by ButterflyPACK.
        set(BUILD_MATGEN OFF CACHE BOOL "" FORCE)
        set(DYNAMIC_ARCH "${KIRCH_OPENBLAS_DYNAMIC_ARCH}" CACHE BOOL "" FORCE)
        set(USE_THREAD "${KIRCH_OPENBLAS_USE_THREADS}" CACHE BOOL "" FORCE)
        set(USE_OPENMP OFF CACHE BOOL "" FORCE)
        # ButterflyPACK invokes BLAS from several OpenMP workers.  A
        # single-threaded OpenBLAS build still needs allocator locking when it
        # is called concurrently; otherwise its shared buffer pool can report
        # "Bad memory unallocation" and corrupt a ButterflyPACK MVP with NaNs.
        set(USE_LOCKING ON CACHE BOOL "" FORCE)
        # OpenBLAS kernels use GNU/Clang inline assembly (`asm`) in many
        # generated object-library targets.  Keep strict C11 for project code,
        # but let every target created by the bundled subdirectory use GNU C
        # extensions (setting this only on openblas_static is too late because
        # the failing kernels are compiled in separate OBJECT libraries).
        set(_kirch_saved_c_extensions "${CMAKE_C_EXTENSIONS}")
        set(CMAKE_C_EXTENSIONS ON)
        add_subdirectory("${PROJECT_SOURCE_DIR}/OpenBLAS"
                         "${PROJECT_BINARY_DIR}/_deps/OpenBLAS")
        set(CMAKE_C_EXTENSIONS "${_kirch_saved_c_extensions}")
        unset(_kirch_saved_c_extensions)
        if(NOT TARGET openblas_static)
            message(FATAL_ERROR "Bundled OpenBLAS did not create openblas_static")
        endif()
        add_library(kirch_blas INTERFACE)
        target_link_libraries(kirch_blas INTERFACE openblas_static)
        add_library(Kirch::BLAS ALIAS kirch_blas)
    elseif(KIRCH_BLAS_PROVIDER STREQUAL "SYSTEM")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        add_library(kirch_blas INTERFACE)
        target_link_libraries(kirch_blas INTERFACE
            ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        add_library(Kirch::BLAS ALIAS kirch_blas)
    else()
        message(FATAL_ERROR
            "KIRCH_BLAS_PROVIDER must be BUNDLED or SYSTEM, got: ${KIRCH_BLAS_PROVIDER}")
    endif()

    # Translate the parent project's option names to the names used by the
    # bundled ButterflyPACK project.  Adding the subdirectory here is what
    # creates ButterflyPACK::float/ButterflyPACK::double; without it
    # SERECKIRCH would compile its adapter without any wrapper declarations.
    set(enable_mpi "${KIRCH_ENABLE_MPI}" CACHE BOOL "" FORCE)
    set(enable_openmp "${KIRCH_ENABLE_OPENMP}" CACHE BOOL "" FORCE)
    set(BPACK_ENABLE_FLOAT "${KIRCH_BPACK_ENABLE_FLOAT}" CACHE BOOL "" FORCE)
    set(BPACK_ENABLE_DOUBLE "${KIRCH_BPACK_ENABLE_DOUBLE}" CACHE BOOL "" FORCE)
    set(BPACK_REGENERATE_PRECISIONS
        "${KIRCH_REGENERATE_BUTTERFLYPACK}" CACHE BOOL "" FORCE)
    set(BPACK_SUPPRESS_WARNINGS
        "${KIRCH_SUPPRESS_THIRD_PARTY_WARNINGS}" CACHE BOOL "" FORCE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/ButterflyPACK"
                     "${PROJECT_BINARY_DIR}/_deps/ButterflyPACK")
endif()
