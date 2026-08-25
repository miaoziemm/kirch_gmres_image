add_library(kirch_eigen INTERFACE)
add_library(Kirch::eigen ALIAS kirch_eigen)
target_include_directories(kirch_eigen SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/eigen")
set(KIRCH_EIGEN_VERSION "bundled")
