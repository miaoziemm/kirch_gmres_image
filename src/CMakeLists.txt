# Layered Huygens-sweeping prototype programs do not all require ButterflyPACK.
foreach(_target IN ITEMS
        huygens_block_info
        travel_time_solver
        wave_frequency_direct
        travel_time_solver_oneway
        wave_frequency_direct_oneway
        wave_frequency_bf_oneway
        frequency_kirchhoff_imaging_direct
        frequency_kirchhoff_imaging_bf)
    set(_source "${_target}.cpp")
    if(_target STREQUAL "frequency_kirchhoff_imaging_bf")
        set(_source "frequency_kirchhoff_imaging_bf_lowmem.cpp")
    endif()
    add_executable(${_target} ${_source})
    target_link_libraries(${_target} PRIVATE Kirch::reckirch)
    target_compile_features(${_target} PRIVATE cxx_std_17)
    set_target_properties(${_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()

# Standalone, double-complex translation of the MATLAB validation script
# test_hybrid_rayphase_residual_gmres_marmousi.m.  It deliberately does not
# depend on ButterflyPACK, the repository GMRES preconditioners, or RSF I/O:
# the MATLAB input is the raw float32 payload (the .rsf@ file).
add_executable(matlab_hybrid_rayphase_gmres
    matlab_hybrid_rayphase_gmres.cpp)
target_link_libraries(matlab_hybrid_rayphase_gmres PRIVATE Kirch::eigen)
target_compile_features(matlab_hybrid_rayphase_gmres PRIVATE cxx_std_17)
set_target_properties(matlab_hybrid_rayphase_gmres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Four-block recursive initial field for the same validation experiment.  It
# uses the in-repository eFMM solver and phase-aware bf1d factorization, so it
# remains available when the optional Fortran ButterflyPACK backend is absent.
# The original matlab_hybrid_rayphase_gmres.cpp is included (with its main
# renamed) to keep the Helmholtz operator and all iterative comparisons exactly
# shared while leaving that source file untouched.
add_executable(matlab_hybrid_recursive_kirchhoff_gmres
    matlab_hybrid_recursive_kirchhoff_gmres.cpp)
target_link_libraries(matlab_hybrid_recursive_kirchhoff_gmres
    PRIVATE Kirch::eigen Kirch::reckirch)
target_compile_features(matlab_hybrid_recursive_kirchhoff_gmres PRIVATE cxx_std_17)
set_target_properties(matlab_hybrid_recursive_kirchhoff_gmres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float)
    add_executable(wave_frequency_bf wave_frequency_bf.cpp)
    target_link_libraries(wave_frequency_bf PRIVATE Kirch::reckirch)
    target_compile_features(wave_frequency_bf PRIVATE cxx_std_17)
    set_target_properties(wave_frequency_bf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

else()
    message(STATUS "Skipping wave_frequency_bf: complex-float ButterflyPACK is unavailable")
endif()

if(KIRCH_ENABLE_BUTTERFLYPACK)
    if(NOT TARGET ButterflyPACK::float AND NOT TARGET ButterflyPACK::double)
        message(FATAL_ERROR
            "kirchhoff_butterfly_rsf requires at least one ButterflyPACK precision")
    endif()

    add_executable(kirchhoff_butterfly_rsf
        kirchhoff_butterfly_rsf.cpp
    )

    target_link_libraries(kirchhoff_butterfly_rsf
        PRIVATE
            Kirch::reckirch
    )

    target_compile_features(kirchhoff_butterfly_rsf PRIVATE cxx_std_17)

    set_target_properties(kirchhoff_butterfly_rsf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
else()
    message(STATUS "Skipping kirchhoff_butterfly_rsf because ButterflyPACK is disabled")
endif()

# The SciPy-compatible correction path uses SuperLU ILUTP as the
# preconditioner and Eigen's restarted GMRES as the iterative solver.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float
   AND TARGET Kirch::SuperLU)
    add_executable(wave_frequency_bf_oneway_gmres
        wave_frequency_bf_oneway_gmres.cpp)
    target_link_libraries(wave_frequency_bf_oneway_gmres
        PRIVATE
            Kirch::reckirch
            Kirch::selinear
            Kirch::SuperLU)
    target_compile_features(wave_frequency_bf_oneway_gmres PRIVATE cxx_std_17)
    set_target_properties(wave_frequency_bf_oneway_gmres PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    message(STATUS
        "Skipping wave_frequency_bf_oneway_gmres: ButterflyPACK float or Kirch::SuperLU is unavailable")
endif()

# Standalone single-frequency source-wavefield path derived from
# frequency_kirchhoff_imaging_bf_gmres: the first block uses the same
# homogeneous phase-shift launch and later blocks use rectangular
# ButterflyPACK, but no receiver wavefield, GMRES correction, or imaging
# arrays are allocated.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float)
    add_executable(frequency_kirchhoff_wavefield_bf
        frequency_kirchhoff_wavefield_bf.cpp)
    target_link_libraries(frequency_kirchhoff_wavefield_bf
        PRIVATE Kirch::reckirch)
    target_compile_features(frequency_kirchhoff_wavefield_bf PRIVATE cxx_std_17)
    set_target_properties(frequency_kirchhoff_wavefield_bf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    message(STATUS
        "Skipping frequency_kirchhoff_wavefield_bf: ButterflyPACK float is unavailable")
endif()

# Clean source-only extraction of the production
# frequency_kirchhoff_imaging_bf_gmres propagation path.  It keeps the
# production Ricker source spectrum, first-block phase-shift launch,
# full-block rectangular ButterflyPACK propagation, and Scheme-A source
# overlap coupling, while allocating no receiver field, GMRES system,
# image, or illumination array.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float)
    add_executable(frequency_kirchhoff_source_wavefield_bf
        frequency_kirchhoff_source_wavefield_bf.cpp)
    target_link_libraries(frequency_kirchhoff_source_wavefield_bf
        PRIVATE Kirch::reckirch)
    target_compile_features(frequency_kirchhoff_source_wavefield_bf PRIVATE cxx_std_17)
    set_target_properties(frequency_kirchhoff_source_wavefield_bf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Production imaging path using the MATLAB Helmholtz/GMRES correction:
# homogeneous phase shift in block one, phase-aware ButterflyPACK Kirchhoff
# initialization in later blocks, then exactly three GMRES outer cycles.
add_executable(frequency_kirchhoff_imaging_bf_gmres
    frequency_kirchhoff_imaging_bf_gmres.cpp)
target_link_libraries(frequency_kirchhoff_imaging_bf_gmres
    PRIVATE Kirch::reckirch Kirch::eigen)
target_compile_features(frequency_kirchhoff_imaging_bf_gmres PRIVATE cxx_std_17)
set_target_properties(frequency_kirchhoff_imaging_bf_gmres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Global correction variant.  It reuses the production ButterflyPACK receiver
# propagation from frequency_kirchhoff_imaging_bf_gmres, but compiles the same
# source with a dedicated global-mode selector.  The source wavefield is built
# from eFMM A*exp(-i*omega*T); only the receiver uses recursive Kirchhoff.
# After the complete receiver field is assembled, source and receiver are
# corrected on the full model with shifted-Laplacian SuperLU-ILUT + GMRES.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float
   AND TARGET Kirch::SuperLU)
    add_executable(frequency_kirchhoff_imaging_bf_global_gmres
        frequency_kirchhoff_imaging_bf_gmres.cpp)
    target_compile_definitions(frequency_kirchhoff_imaging_bf_global_gmres
        PRIVATE KIRCH_GLOBAL_PRECONDITIONED_GMRES=1)
    target_link_libraries(frequency_kirchhoff_imaging_bf_global_gmres
        PRIVATE
            Kirch::reckirch
            Kirch::eigen
            Kirch::SuperLU)
    target_compile_features(
        frequency_kirchhoff_imaging_bf_global_gmres PRIVATE cxx_std_17)
    set_target_properties(frequency_kirchhoff_imaging_bf_global_gmres PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    message(STATUS
        "Skipping frequency_kirchhoff_imaging_bf_global_gmres: ButterflyPACK float or Kirch::SuperLU is unavailable")
endif()

# Direct-Kirchhoff benchmark using the same Scheme-A + GMRES implementation
# as frequency_kirchhoff_imaging_bf_gmres.  The tiny wrapper source defines a
# compile-time backend selector before including the shared implementation, so
# the two executables cannot silently diverge in seam coupling or correction.
add_executable(frequency_kirchhoff_imaging_direct_gmres
    frequency_kirchhoff_imaging_direct_gmres.cpp)
target_link_libraries(frequency_kirchhoff_imaging_direct_gmres
    PRIVATE Kirch::reckirch Kirch::eigen)
target_compile_features(frequency_kirchhoff_imaging_direct_gmres PRIVATE cxx_std_17)
set_target_properties(frequency_kirchhoff_imaging_direct_gmres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Interleaved ordering experiment:
# C1 -> K2 -> C2 -> K3 -> C3 -> ... -> K5 -> C5.
# This target is independent of the existing full-BF-then-GMRES executable.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float
   AND TARGET Kirch::SuperLU)
    add_executable(wave_frequency_bf_oneway_gmres_interleaved
        wave_frequency_bf_oneway_gmres_interleaved.cpp)
    target_link_libraries(wave_frequency_bf_oneway_gmres_interleaved
        PRIVATE
            Kirch::reckirch
            Kirch::selinear
            Kirch::SuperLU)
    target_compile_features(
        wave_frequency_bf_oneway_gmres_interleaved PRIVATE cxx_std_17)
    set_target_properties(wave_frequency_bf_oneway_gmres_interleaved PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    message(STATUS
        "Skipping wave_frequency_bf_oneway_gmres_interleaved: ButterflyPACK float or Kirch::SuperLU is unavailable")
endif()
