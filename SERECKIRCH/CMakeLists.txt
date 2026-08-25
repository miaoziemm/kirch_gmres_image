set(RECKIRCH_HEADERS
    include/bf1d.h
    include/se_blas.hpp
    include/se_fftw.hpp
    include/se_butterfly.hpp
    include/decart.h
    include/efmm.h
    include/frequency_kirchhoff_imaging_common.hpp
    include/global_bf_imaging_workflow.hpp
    include/huygens_cli.hpp
    include/huygens_sweep.hpp
    include/eikods.h
    include/kirdat.h
    include/kirmig.h
    include/mutter.h
    include/neighbors.h
    include/pqueue.h
    include/se_reckirch.h
    include/tinterpsr.h
    include/tpow.h
)

set(RECKIRCH_SOURCES
    src/bf1d.cpp
    src/se_blas.cpp
    src/se_fftw.cpp
    src/decart.cpp
    src/efmm.cpp
    src/huygens_sweep.cpp
    src/eikods.cpp
    src/kirdat.cpp
    src/kirmig.cpp
    src/mutter.cpp
    src/neighbors.cpp
    src/pqueue.cpp
    src/tinterpsr.cpp
    src/tpow.cpp
)

# These are the dedicated ButterflyPACK/Kirchhoff wrappers prepared for the
# multi-frequency interface. They are included automatically when copied into
# SERECKIRCH, while the current repository can still build without them.
set(_reckirch_optional_sources
    src/butterfly.cpp
    src/frequency_bank.cpp
    src/operator.cpp
    src/traveltime.cpp
)
set(_reckirch_optional_headers
    include/kirchhoff/butterfly.hpp
    include/kirchhoff/frequency_bank.hpp
    include/kirchhoff/operator.hpp
    include/kirchhoff/traveltime.hpp
)

foreach(_file IN LISTS _reckirch_optional_sources)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
        list(APPEND RECKIRCH_SOURCES "${_file}")
    endif()
endforeach()
foreach(_file IN LISTS _reckirch_optional_headers)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
        list(APPEND RECKIRCH_HEADERS "${_file}")
    endif()
endforeach()

add_library(reckirch STATIC ${RECKIRCH_SOURCES} ${RECKIRCH_HEADERS})
add_library(Kirch::reckirch ALIAS reckirch)

set_target_properties(reckirch PROPERTIES
    OUTPUT_NAME reckirch
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(reckirch
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SERECKIRCH/include>
)

target_link_libraries(reckirch
    PUBLIC
        Kirch::sebasic
        Kirch::sefs
        Kirch::project_options
        Kirch::parallel
        ${KIRCH_FFTW_FLOAT_TARGET}
        ${KIRCH_FFTW_DOUBLE_TARGET}
)

target_compile_definitions(reckirch PUBLIC
    KIRCH_HAS_FFTW_FLOAT=1
    KIRCH_HAS_FFTW_DOUBLE=1
)

if(KIRCH_FFTW_PROVIDER STREQUAL "BUNDLED")
    target_include_directories(reckirch SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/fftw/api>)
endif()

if(KIRCH_ENABLE_BUTTERFLYPACK AND
   (TARGET ButterflyPACK::float OR TARGET ButterflyPACK::double))
    target_sources(reckirch PRIVATE src/se_butterfly.cpp)
endif()

if(TARGET Kirch::BLAS)
    target_link_libraries(reckirch PUBLIC Kirch::BLAS)
endif()

set(_reckirch_has_butterflypack FALSE)

if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::float)
    target_link_libraries(reckirch PUBLIC ButterflyPACK::float)
    target_compile_definitions(reckirch PUBLIC
        KIRCH_HAS_BUTTERFLYPACK_FLOAT=1)
    set(_reckirch_has_butterflypack TRUE)
endif()

if(KIRCH_ENABLE_BUTTERFLYPACK AND TARGET ButterflyPACK::double)
    target_link_libraries(reckirch PUBLIC ButterflyPACK::double)
    target_compile_definitions(reckirch PUBLIC
        KIRCH_HAS_BUTTERFLYPACK_DOUBLE=1)
    set(_reckirch_has_butterflypack TRUE)
endif()

if(_reckirch_has_butterflypack)
    target_compile_definitions(reckirch PUBLIC KIRCH_HAS_BUTTERFLYPACK=1)
endif()

if(KIRCH_ENABLE_INSTALL)
    install(DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SERECKIRCH/include")
endif()
