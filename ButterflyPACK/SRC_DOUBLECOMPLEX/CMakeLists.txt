include_directories(${ButterflyPACK_SOURCE_DIR}/MPI_DUMMY)

set(headers
    zBPACK_wrapper.h
    ButterflyPACK_config.fi 	zButterflyPACK_config.fi
    # ${PROJECT_BINARY_DIR}/ButterflyPACK_config.h
)

set(sources
zMAGMA_offset.c
zMAGMA_utilities.f90
zLAPACK_dtrevc3mod.f
zLAPACK_ztrevc3mod.f
zLAPACK_strevc3mod.f
zLAPACK_ctrevc3mod.f
zLAPACK_dgeqp3mod.f
zLAPACK_zgeqp3mod.f
zLAPACK_sgeqp3mod.f
zLAPACK_cgeqp3mod.f
zLAPACK_dgetrfmod.f
zLAPACK_dgetrf2mod.f
zLAPACK_zgetrfmod.f
zLAPACK_zgetrf2mod.f
zLAPACK_sgetrfmod.f
zLAPACK_sgetrf2mod.f
zLAPACK_cgetrfmod.f
zLAPACK_cgetrf2mod.f
zMISC_linkedlist.f90
zMISC_DenseLA.f90
zBPACK_defs.f90
zMISC_utilities.f90
zBplus_utilities.f90
zBPACK_utilities.f90
zBplus_pack_unpack_for_MPI.f90
zBPACK_structure.f90
zBplus_randomized.f90
zBplus_compress.f90
zBplus_factor.f90
zBPACK_solve_mul.f90
zBPACK_constr.f90
zBPACK_randomized.f90
zBPACK_factor.f90
zBPACK_wrapper.f90
zC_BPACK_wrapper.cpp
)

if (enable_mpi)
  list(APPEND sources zSCALAPACK_pdgeqpfmod.f zSCALAPACK_pzgeqpfmod.f zSCALAPACK_psgeqpfmod.f zSCALAPACK_pcgeqpfmod.f zSCALAPACK_pdgetrfmod.f zSCALAPACK_pzgetrfmod.f zSCALAPACK_psgetrfmod.f zSCALAPACK_pcgetrfmod.f zSCALAPACK_pdgetf2mod.f zSCALAPACK_pzgetf2mod.f zSCALAPACK_psgetf2mod.f zSCALAPACK_pcgetf2mod.f)
endif ()


add_library(zbutterflypack ${sources} ${HEADERS})

if (enable_mpi)
target_link_libraries(zbutterflypack
                       ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${SCALAPACK_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
else()
target_link_libraries(zbutterflypack mpi_dummy ${BLAS_LIB} ${LAPACK_LIB} ${ZFP_LIB} ${ARPACK_LIB} ${MAGMA_LIB} m)
endif()
                      set_target_properties(zbutterflypack PROPERTIES
                      VERSION ${PROJECT_VERSION} SOVERSION ${VERSION_MAJOR} COMPILE_FLAGS "-DDAT=0"
)

target_include_directories(zbutterflypack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)


# ---- Python extension module ----
if (enable_python)
  add_library(zbutterflypack_python SHARED zPy_BPACK_wrapper.cpp)
  set_target_properties(zbutterflypack_python PROPERTIES PREFIX "")
  add_dependencies(zbutterflypack_python zbutterflypack)
  target_link_libraries(zbutterflypack_python
    PUBLIC
      zbutterflypack
      pybind11::module
  )
  target_include_directories(zbutterflypack_python PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
  )
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS zbutterflypack EXPORT MYBPACKTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (enable_python)
  install(TARGETS zbutterflypack_python EXPORT MYBPACKTargets
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
      zPy_BPACK_wrapper.py
      zPy_BPACK_worker.py
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
  add_custom_target(z_py_bpack ALL
      DEPENDS ${PY_SCRIPT_OUTPUTS}
  )
  install(
      FILES ${PY_SCRIPTS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()