# Executables and their application-specific helper headers are maintained
# together in this directory; the build does not depend on the reference
# implementations under ref/.

function(kirch_add_application target)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

# Model partitioning and eFMM travel-time-table generation.
kirch_add_application(huygens_block_info huygens_block_info.cpp)
target_link_libraries(huygens_block_info PRIVATE Kirch::reckirch)

kirch_add_application(travel_time_solver travel_time_solver.cpp)
target_link_libraries(travel_time_solver PRIVATE Kirch::reckirch)

# Time-domain acoustic forward modelling.  SEWAVE is not a standalone CMake
# library in this repository, so compile its implementation into this target.
kirch_add_application(sewave2d_forward
    sewave2d_forward.cpp
    "${PROJECT_SOURCE_DIR}/SEWAVE/sewave2d.cpp")
target_link_libraries(sewave2d_forward
    PRIVATE Kirch::sebasic Kirch::sefs ${KIRCH_FFTW_FLOAT_TARGET})

foreach(_wave_utility IN ITEMS smooth_velocity_model mute_direct_wave)
    kirch_add_application(${_wave_utility}
        ${_wave_utility}.cpp
        "${PROJECT_SOURCE_DIR}/SEWAVE/sewave2d.cpp")
    target_link_libraries(${_wave_utility}
        PRIVATE Kirch::sebasic Kirch::sefs ${KIRCH_FFTW_FLOAT_TARGET})
endforeach()

# The production imaging executable constructs complex-float ButterflyPACK
# matrices, and therefore must not be offered by configurations which disable
# that backend.
if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float)
    # This target intentionally uses the reference restarted-GMRES algorithm,
    # which only requires Eigen and does not use the production SuperLU path.
    kirch_add_application(f_image_rtm_kirch f_image_rtm_kirch.cpp)
    target_link_libraries(f_image_rtm_kirch
        PRIVATE Kirch::reckirch Kirch::selinear Kirch::eigen)
else()
    message(STATUS
        "Skipping f_image_rtm_kirch: ButterflyPACK float is unavailable")
endif()

if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float
   AND TARGET Kirch::SuperLU)
    kirch_add_application(frequency_kirchhoff_imaging_bf_gmres
        frequency_kirchhoff_imaging_bf_gmres.cpp
        ${PROJECT_SOURCE_DIR}/SERECKIRCH/src/global_bf_imaging_workflow.cpp)
    kirch_add_application(f_image_rtm_kirch_multishot
        f_image_rtm_kirch_multishot.cpp
        ${PROJECT_SOURCE_DIR}/SERECKIRCH/src/global_bf_imaging_workflow.cpp)
    target_link_libraries(frequency_kirchhoff_imaging_bf_gmres
        PRIVATE Kirch::reckirch Kirch::eigen Kirch::SuperLU)
    target_link_libraries(f_image_rtm_kirch_multishot
        PRIVATE Kirch::reckirch Kirch::eigen Kirch::SuperLU)

else()
    message(STATUS
        "Skipping frequency_kirchhoff_imaging_bf_gmres: ButterflyPACK float or SuperLU is unavailable")
endif()
