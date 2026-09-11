# src/CMakeLists.txt
#
# Compile the applications currently used in src/.

function(kirch_add_application target)

    add_executable(
        ${target}
        ${ARGN}
    )

    target_compile_features(
        ${target}
        PRIVATE
            cxx_std_17
    )

    set_target_properties(
        ${target}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin"
    )

endfunction()


# ============================================================
# Huygens block information
# ============================================================

kirch_add_application(
    huygens_block_info
    huygens_block_info.cpp
)

target_link_libraries(
    huygens_block_info
    PRIVATE
        Kirch::reckirch
)


# ============================================================
# Travel-time solver
# ============================================================

kirch_add_application(
    travel_time_solver
    travel_time_solver.cpp
)

target_link_libraries(
    travel_time_solver
    PRIVATE
        Kirch::reckirch
)


# ============================================================
# Shared SEWAVE implementation
# ============================================================

add_library(
    kirch_sewave2d
    STATIC
        "${PROJECT_SOURCE_DIR}/SEWAVE/sewave2d.cpp"
)

target_compile_features(
    kirch_sewave2d
    PUBLIC
        cxx_std_17
)

target_include_directories(
    kirch_sewave2d
    PUBLIC
        "${PROJECT_SOURCE_DIR}/SEWAVE"
)

target_link_libraries(
    kirch_sewave2d
    PUBLIC
        Kirch::sebasic
        Kirch::sefs
        ${KIRCH_FFTW_FLOAT_TARGET}
)


# ============================================================
# Time-domain acoustic forward modeling
# ============================================================

kirch_add_application(
    sewave2d_forward
    sewave2d_forward.cpp
)

target_link_libraries(
    sewave2d_forward
    PRIVATE
        kirch_sewave2d
)


# ============================================================
# Velocity smoothing
# ============================================================

kirch_add_application(
    smooth_velocity_model
    smooth_velocity_model.cpp
)

target_link_libraries(
    smooth_velocity_model
    PRIVATE
        kirch_sewave2d
)


# ============================================================
# Direct-wave mute
# ============================================================

kirch_add_application(
    mute_direct_wave
    mute_direct_wave.cpp
)

target_link_libraries(
    mute_direct_wave
    PRIVATE
        kirch_sewave2d
)


# ============================================================
# ButterflyPACK Kirchhoff test
# ============================================================

kirch_add_application(
    test_bf_kir
    test_bf_kir.cpp
)

target_link_libraries(
    test_bf_kir
    PRIVATE
        Kirch::reckirch
        Kirch::sefs
)