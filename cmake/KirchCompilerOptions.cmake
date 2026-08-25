add_library(kirch_project_options INTERFACE)
add_library(Kirch::project_options ALIAS kirch_project_options)

if(KIRCH_WARNINGS_AS_ERRORS)
    target_compile_options(kirch_project_options INTERFACE
        $<$<COMPILE_LANG_AND_ID:C,CXX,GNU,Clang,AppleClang>:-Werror>)
endif()

add_library(kirch_parallel INTERFACE)
add_library(Kirch::parallel ALIAS kirch_parallel)
if(KIRCH_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    target_link_libraries(kirch_parallel INTERFACE
        OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()
