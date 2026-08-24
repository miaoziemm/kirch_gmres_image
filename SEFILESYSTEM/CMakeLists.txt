set(SEFS_HEADERS
    include/se_coordinate_system_par.h
    include/se_fs.h
    include/se_fs_io.h
    include/se_fs_segy.h
    include/se_fs_segy_bin_types.h
    include/se_fs_segy_header.h
    include/se_fs_segy_index.h
    include/se_fs_segy_trchead_eval.h
    include/se_fs_sep.h
    include/se_fs_sep_fake.h
    include/se_fs_struct.h
    include/se_io.h
    include/se_jcs.h
    include/se_module_par_desc_sep.h
    include/se_ode_cfg_sep.h
    include/se_par_getset.h
    include/se_par_sep.h
    include/se_uri.h
    include/se_vel_cfg_sep.h
    include/sesi_map.h
)

set(SEFS_SOURCES
    src/se_coordinate_system_par.cpp
    src/se_fs_io.cpp
    src/se_fs_segy_bin_types.cpp
    src/se_fs_segy_data.cpp
    src/se_fs_segy_header.cpp
    src/se_fs_segy_index.cpp
    src/se_fs_segy_io.cpp
    src/se_fs_segy_trchead_eval.cpp
    src/se_fs_segy_util.cpp
    src/se_fs_sep.cpp
    src/se_io.cpp
    src/se_jcs.cpp
    src/se_module_par_desc_sep.cpp
    src/se_ode_cfg_sep.cpp
    src/se_par_getset.cpp
    src/se_par_sep.cpp
    src/se_uri.cpp
    src/se_vel_cfg_sep.cpp
    src/sesi_map.cpp
)

add_library(sefs STATIC ${SEFS_SOURCES} ${SEFS_HEADERS})
add_library(Kirch::sefs ALIAS sefs)

set_target_properties(sefs PROPERTIES
    OUTPUT_NAME sefs
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(sefs
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SEFILESYSTEM/include>
)

target_link_libraries(sefs
    PUBLIC
        Kirch::sebasic
        Kirch::project_options
        Kirch::parallel
)

if(KIRCH_ENABLE_INSTALL)
    install(DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SEFILESYSTEM/include")
endif()
