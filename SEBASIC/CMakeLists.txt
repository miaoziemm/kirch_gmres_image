set(SEBASIC_HEADERS
    include/json.hpp
    include/se_agc.h
    include/se_alloc.h
    include/se_allpass.h
    include/se_apfilt.h
    include/se_array.h
    include/se_array_bit.h
    include/se_assert.h
    include/se_bandpass.h
    include/se_basic.h
    include/se_basic_math.h
    include/se_byte.h
    include/se_coh_func.h
    include/se_coordinate_system.h
    include/se_exe.h
    include/se_expression.h
    include/se_gb.h
    include/se_geometry.h
    include/se_green_func.h
    include/se_grid.h
    include/se_hash.h
    include/se_horizon_model.h
    include/se_interpolation.h
    include/se_log.h
    include/se_macro.h
    include/se_math_physics.h
    include/se_max_factor.h
    include/se_minmax.h
    include/se_module_par_desc.h
    include/se_ode_cfg.h
    include/se_par.h
    include/se_pick_file.h
    include/se_quantiler.h
    include/se_ray.h
    include/se_ray_trace_rk.h
    include/se_return_code.h
    include/se_rsf.h
    include/se_socket.h
    include/se_sort.h
    include/se_sparse_interpolation.h
    include/se_stack.h
    include/se_struct_inter.h
    include/se_su_fft.h
    include/se_su_par.h
    include/se_sysinfo.h
    include/se_thread.h
    include/se_timer.h
    include/se_trace_gb.h
    include/se_trace_ray.h
    include/se_type.h
    include/se_util.h
    include/se_vconv.h
    include/se_vel.h
    include/se_vel_cfg.h
    include/se_version.h
    include/se_wavelet.h
    include/sqlite3.h
)

set(SEBASIC_SOURCES
    src/se_agc.cpp
    src/se_alloc.cpp
    src/se_allpass.cpp
    src/se_apfilt.cpp
    src/se_array.cpp
    src/se_array_bit.cpp
    src/se_assert.cpp
    src/se_bandpass.cpp
    src/se_basic_math.cpp
    src/se_byte.cpp
    src/se_coh_func.cpp
    src/se_exe.cpp
    src/se_expression.cpp
    src/se_geometry.cpp
    src/se_green_func.cpp
    src/se_hash.cpp
    src/se_horizon_model.cpp
    src/se_interpolation.cpp
    src/se_log.cpp
    src/se_math_physics.cpp
    src/se_max_factor.cpp
    src/se_module_par_desc.cpp
    src/se_ode_cfg.cpp
    src/se_pick_file.cpp
    src/se_quantiler.cpp
    src/se_ray.cpp
    src/se_ray_trace2d_rk4.cpp
    src/se_ray_trace3d_rk4.cpp
    src/se_socket.cpp
    src/se_sort.cpp
    src/se_sparse_interpolation.cpp
    src/se_stack.cpp
    src/se_su_fft.cpp
    src/se_su_par.cpp
    src/se_sysinfo.cpp
    src/se_timer.cpp
    src/se_trace_gb.cpp
    src/se_trace_ray.cpp
    src/se_util.cpp
    src/se_vconv.cpp
    src/se_vel.cpp
    src/se_vel_cfg.cpp
    src/se_wavelet.cpp
    src/sqlite3.c
)

add_library(sebasic STATIC ${SEBASIC_SOURCES} ${SEBASIC_HEADERS})
add_library(Kirch::sebasic ALIAS sebasic)

set_target_properties(sebasic PROPERTIES
    OUTPUT_NAME sebasic
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(sebasic
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SEBASIC/include>
)

target_link_libraries(sebasic
    PUBLIC
        Kirch::selinear
        Kirch::project_options
        Kirch::parallel
        Threads::Threads
)

# SQLite is third-party C code. Suppress its warnings only, not project warnings.
set_source_files_properties(src/sqlite3.c PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-w>"
)

if(KIRCH_ENABLE_INSTALL)
    install(DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SEBASIC/include")
endif()
