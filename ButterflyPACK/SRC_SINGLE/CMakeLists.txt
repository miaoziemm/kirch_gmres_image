include_directories(${ButterflyPACK_SOURCE_DIR}/MPI_DUMMY)

set(headers
    sBPACK_wrapper.h
    ButterflyPACK_config.fi 	sButterflyPACK_config.fi
    # ${PROJECT_BINARY_DIR}/ButterflyPACK_config.h
)

set(sources
sMAGMA_offset.c
sMAGMA_utilities.f90
sLAPACK_dtrevc3mod.f
sLAPACK_ztrevc3mod.f
sLAPACK_strevc3mod.f
sLAPACK_ctrevc3mod.f
sLAPACK_dgeqp3mod.f
sLAPACK_zgeqp3mod.f
sLAPACK_sgeqp3mod.f
sLAPACK_cgeqp3mod.f
sLAPACK_dgetrfmod.f
sLAPACK_dgetrf2mod.f
sLAPACK_zgetrfmod.f
sLAPACK_zgetrf2mod.f
sLAPACK_sgetrfmod.f
sLAPACK_sgetrf2mod.f
sLAPACK_cgetrfmod.f
sLAPACK_cgetrf2mod.f
sMISC_linkedlist.f90
sMISC_DenseLA.f90
sBPACK_defs.f90
sMISC_utilities.f90
sBplus_utilities.f90
sBPACK_utilities.f90
sBplus_pack_unpack_for_MPI.f90
sBPACK_structure.f90
sBplus_randomized.f90
sBplus_compress.f90
sBplus_factor.f90
sBPACK_solve_mul.f90
sBPACK_constr.f90
sBPACK_randomized.f90
sBPACK_factor.f90
sBPACK_wrapper.f90
sC_BPACK_wrapper.cpp
)

if (enable_mpi)
  list(APPEND sources sSCALAPACK_pdgeqpfmod.f sSCALAPACK_pzgeqpfmod.f sSCALAPACK_psgeqpfmod.f sSCALAPACK_pcgeqpfmod.f sSCALAPACK_pdgetrfmod.f sSCALAPACK_pzgetrfmod.f sSCALAPACK_psgetrfmod.f sSCALAPACK_pcgetrfmod.f sSCALAPACK_pdgetf2mod.f sSCALAPACK_pzgetf2mod.f sSCALAPACK_psgetf2mod.f sSCALAPACK_pcgetf2mod.f)
endif ()


add_library(sbutterflypack ${sources} ${HEADERS})

if (enable_mpi)
target_link_libraries(sbutterflypack
                       ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${SCALAPACK_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
else()
target_link_libraries(sbutterflypack mpi_dummy ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
endif()
                      set_target_properties(sbutterflypack PROPERTIES
                      VERSION ${PROJECT_VERSION} SOVERSION ${VERSION_MAJOR} COMPILE_FLAGS "-DDAT=3"
)

target_include_directories(sbutterflypack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)


# ---- Python extension module ----
if (enable_python)
  add_library(sbutterflypack_python SHARED sPy_BPACK_wrapper.cpp)
  set_target_properties(sbutterflypack_python PROPERTIES PREFIX "")
  add_dependencies(sbutterflypack_python sbutterflypack)
  target_link_libraries(sbutterflypack_python
    PUBLIC
      sbutterflypack
      pybind11::module
  )
  target_include_directories(sbutterflypack_python PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
  )
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS sbutterflypack EXPORT MYBPACKTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (enable_python)
  install(TARGETS sbutterflypack_python EXPORT MYBPACKTargets
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
      sPy_BPACK_wrapper.py
      sPy_BPACK_worker.py
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
  add_custom_target(s_py_bpack ALL
      DEPENDS ${PY_SCRIPT_OUTPUTS}
  )
  install(
      FILES ${PY_SCRIPTS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()