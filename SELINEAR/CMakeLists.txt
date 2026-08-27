set(SELINEAR_HEADERS
    include/global_preconditioned_gmres.hpp
    include/reference_global_gmres.hpp
    include/se_eigen.hpp
    include/se_eigen_solver.hpp
    include/se_eigen_solver_impl.hpp
    include/se_eigen_complex_dispatch.hpp
)

set(SELINEAR_SOURCES
    src/se_eigen_solver.cpp
)

add_library(selinear STATIC ${SELINEAR_SOURCES} ${SELINEAR_HEADERS})
add_library(Kirch::selinear ALIAS selinear)

set_target_properties(selinear PROPERTIES
    OUTPUT_NAME selinear
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(selinear
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SELINEAR/include>
)

target_link_libraries(selinear
    PUBLIC
        Kirch::project_options
        Kirch::eigen
)

target_compile_definitions(selinear PUBLIC KIRCH_HAS_EIGEN=1)

if(BUILD_TESTING)
    add_executable(se_eigen_solver_test
        tests/se_eigen_solver_test.cpp)
    target_link_libraries(se_eigen_solver_test PRIVATE Kirch::selinear)
    add_test(NAME se_eigen_solver_api COMMAND se_eigen_solver_test)
endif()

if(KIRCH_ENABLE_INSTALL)
    install(DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SELINEAR/include")
endif()
