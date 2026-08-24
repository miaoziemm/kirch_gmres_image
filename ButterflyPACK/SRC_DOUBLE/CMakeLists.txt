include_directories(${ButterflyPACK_SOURCE_DIR}/MPI_DUMMY)

set(headers
    dBPACK_wrapper.h
    ButterflyPACK_config.fi 	dButterflyPACK_config.fi
    # ${PROJECT_BINARY_DIR}/ButterflyPACK_config.h
)

set(sources
dMAGMA_offset.c
dMAGMA_utilities.f90
dLAPACK_dtrevc3mod.f
dLAPACK_ztrevc3mod.f
dLAPACK_strevc3mod.f
dLAPACK_ctrevc3mod.f
dLAPACK_dgeqp3mod.f
dLAPACK_zgeqp3mod.f
dLAPACK_sgeqp3mod.f
dLAPACK_cgeqp3mod.f
dLAPACK_dgetrfmod.f
dLAPACK_dgetrf2mod.f
dLAPACK_zgetrfmod.f
dLAPACK_zgetrf2mod.f
dLAPACK_sgetrfmod.f
dLAPACK_sgetrf2mod.f
dLAPACK_cgetrfmod.f
dLAPACK_cgetrf2mod.f
dMISC_linkedlist.f90
dMISC_DenseLA.f90
dBPACK_defs.f90
dMISC_utilities.f90
dBplus_utilities.f90
dBPACK_utilities.f90
dBplus_pack_unpack_for_MPI.f90
dBPACK_structure.f90
dBplus_randomized.f90
dBplus_compress.f90
dBplus_factor.f90
dBPACK_solve_mul.f90
dBPACK_constr.f90
dBPACK_randomized.f90
dBPACK_factor.f90
dBPACK_wrapper.f90
dC_BPACK_wrapper.cpp
)

if (enable_mpi)
  list(APPEND sources dSCALAPACK_pdgeqpfmod.f dSCALAPACK_pzgeqpfmod.f dSCALAPACK_psgeqpfmod.f dSCALAPACK_pcgeqpfmod.f dSCALAPACK_pdgetrfmod.f dSCALAPACK_pzgetrfmod.f dSCALAPACK_psgetrfmod.f dSCALAPACK_pcgetrfmod.f dSCALAPACK_pdgetf2mod.f dSCALAPACK_pzgetf2mod.f dSCALAPACK_psgetf2mod.f dSCALAPACK_pcgetf2mod.f)
endif ()


add_library(dbutterflypack ${sources} ${HEADERS})

if (enable_mpi)
target_link_libraries(dbutterflypack
                       ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${SCALAPACK_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
else()
target_link_libraries(dbutterflypack mpi_dummy ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
endif()
                      set_target_properties(dbutterflypack PROPERTIES
                      VERSION ${PROJECT_VERSION} SOVERSION ${VERSION_MAJOR} COMPILE_FLAGS "-DDAT=1"
)

target_include_directories(dbutterflypack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)


# ---- Python extension module ----
if (enable_python)
  add_library(dbutterflypack_python SHARED dPy_BPACK_wrapper.cpp)
  set_target_properties(dbutterflypack_python PROPERTIES PREFIX "")
  add_dependencies(dbutterflypack_python dbutterflypack)
  target_link_libraries(dbutterflypack_python
    PUBLIC
      dbutterflypack
      pybind11::module
  )
  target_include_directories(dbutterflypack_python PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
  )
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS dbutterflypack EXPORT MYBPACKTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (enable_python)
  install(TARGETS dbutterflypack_python EXPORT MYBPACKTargets
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
      dPy_BPACK_wrapper.py
      dPy_BPACK_worker.py
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
  add_custom_target(d_py_bpack ALL
      DEPENDS ${PY_SCRIPT_OUTPUTS}
  )
  install(
      FILES ${PY_SCRIPTS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()