include_directories(${ButterflyPACK_SOURCE_DIR}/MPI_DUMMY)

set(headers
    cBPACK_wrapper.h
    ButterflyPACK_config.fi 	cButterflyPACK_config.fi
    # ${PROJECT_BINARY_DIR}/ButterflyPACK_config.h
)

set(sources
cMAGMA_offset.c
cMAGMA_utilities.f90
cLAPACK_dtrevc3mod.f
cLAPACK_ztrevc3mod.f
cLAPACK_strevc3mod.f
cLAPACK_ctrevc3mod.f
cLAPACK_dgeqp3mod.f
cLAPACK_zgeqp3mod.f
cLAPACK_sgeqp3mod.f
cLAPACK_cgeqp3mod.f
cLAPACK_dgetrfmod.f
cLAPACK_dgetrf2mod.f
cLAPACK_zgetrfmod.f
cLAPACK_zgetrf2mod.f
cLAPACK_sgetrfmod.f
cLAPACK_sgetrf2mod.f
cLAPACK_cgetrfmod.f
cLAPACK_cgetrf2mod.f
cMISC_linkedlist.f90
cMISC_DenseLA.f90
cBPACK_defs.f90
cMISC_utilities.f90
cBplus_utilities.f90
cBPACK_utilities.f90
cBplus_pack_unpack_for_MPI.f90
cBPACK_structure.f90
cBplus_randomized.f90
cBplus_compress.f90
cBplus_factor.f90
cBPACK_solve_mul.f90
cBPACK_constr.f90
cBPACK_randomized.f90
cBPACK_factor.f90
cBPACK_wrapper.f90
cC_BPACK_wrapper.cpp
)

if (enable_mpi)
  list(APPEND sources cSCALAPACK_pdgeqpfmod.f cSCALAPACK_pzgeqpfmod.f cSCALAPACK_psgeqpfmod.f cSCALAPACK_pcgeqpfmod.f cSCALAPACK_pdgetrfmod.f cSCALAPACK_pzgetrfmod.f cSCALAPACK_psgetrfmod.f cSCALAPACK_pcgetrfmod.f cSCALAPACK_pdgetf2mod.f cSCALAPACK_pzgetf2mod.f cSCALAPACK_psgetf2mod.f cSCALAPACK_pcgetf2mod.f)
endif ()


add_library(cbutterflypack ${sources} ${HEADERS})

if (enable_mpi)
target_link_libraries(cbutterflypack
                       ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${SCALAPACK_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
else()
target_link_libraries(cbutterflypack mpi_dummy ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
endif()
                      set_target_properties(cbutterflypack PROPERTIES
                      VERSION ${PROJECT_VERSION} SOVERSION ${VERSION_MAJOR} COMPILE_FLAGS "-DDAT=2"
)

target_include_directories(cbutterflypack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)


# ---- Python extension module ----
if (enable_python)
  add_library(cbutterflypack_python SHARED cPy_BPACK_wrapper.cpp)
  set_target_properties(cbutterflypack_python PROPERTIES PREFIX "")
  add_dependencies(cbutterflypack_python cbutterflypack)
  target_link_libraries(cbutterflypack_python
    PUBLIC
      cbutterflypack
      pybind11::module
  )
  target_include_directories(cbutterflypack_python PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
  )
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS cbutterflypack EXPORT MYBPACKTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (enable_python)
  install(TARGETS cbutterflypack_python EXPORT MYBPACKTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()


install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


#SET(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.mod")


if (enable_python)
  set(PY_SCRIPTS
      cPy_BPACK_wrapper.py
      cPy_BPACK_worker.py
  )
  set(PY_SCRIPT_OUTPUTS)
  foreach(pyfile IN LISTS PY_SCRIPTS)
      set(src ${CMAKE_CURRENT_SOURCE_DIR}/${pyfile})
      set(dst ${CMAKE_CURRENT_BINARY_DIR}/${pyfile})

      add_custom_command(
          OUTPUT ${dst}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
          DEPENDS ${src}
          COMMENT "Copying ${pyfile}"
      )

      list(APPEND PY_SCRIPT_OUTPUTS ${dst})
  endforeach()
  add_custom_target(c_py_bpack ALL
      DEPENDS ${PY_SCRIPT_OUTPUTS}
  )
  install(
      FILES ${PY_SCRIPTS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()