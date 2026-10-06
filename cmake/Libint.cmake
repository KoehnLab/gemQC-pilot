# Locate libint2 (C++ API): an installed libint (e.g. Homebrew: max AM 4) or a
# custom build (see docs/developer/libint_build.md), selected via
# -DLibint2_DIR=<prefix>/lib/cmake/libint2.
# Provides GEMQC_LIBINT_TARGET (shared library preferred, static with
# -DGEMQC_LIBINT_STATIC=ON) and GEMQC_LIBINT_RPATH (library directory of a shared
# libint, so that exactly this installation is loaded at run time).

# libint2 needs Eigen3 (header-only). If no installed Eigen3 is found,
# download the headers once into the build tree and provide Eigen3::Eigen.
# (no version in find_package: Eigen's version check rejects other major
#  versions, e.g. 5.x for a request of 3.3; check the minimum instead)
find_package(Eigen3 CONFIG QUIET)
if(TARGET Eigen3::Eigen AND Eigen3_VERSION VERSION_LESS 3.3)
  message(FATAL_ERROR "Eigen3 ${Eigen3_VERSION} found, >= 3.3 required")
endif()
if(NOT TARGET Eigen3::Eigen)
  set(GEMQC_EIGEN_VERSION 3.4.0)
  set(_eigen_root "${CMAKE_BINARY_DIR}/_deps/eigen-${GEMQC_EIGEN_VERSION}")
  if(NOT EXISTS "${_eigen_root}/Eigen/Core")
    set(_eigen_tar "${CMAKE_BINARY_DIR}/_deps/eigen-${GEMQC_EIGEN_VERSION}.tar.gz")
    message(STATUS "Eigen3 not found, downloading Eigen ${GEMQC_EIGEN_VERSION}")
    file(DOWNLOAD
      "https://gitlab.com/libeigen/eigen/-/archive/${GEMQC_EIGEN_VERSION}/eigen-${GEMQC_EIGEN_VERSION}.tar.gz"
      "${_eigen_tar}" STATUS _dl_status TLS_VERIFY ON)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(FATAL_ERROR "Eigen3 download failed: ${_dl_status}. "
        "Install Eigen3 or set Eigen3_DIR to its share/eigen3/cmake directory.")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_eigen_tar}" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_eigen_root}")
  message(STATUS "Using Eigen3 headers from ${_eigen_root}")
endif()

find_package(Libint2 2.7 CONFIG)
if(NOT Libint2_FOUND)
  message(FATAL_ERROR
    "libint2 not found. Install it (e.g. 'brew install libint') or set "
    "Libint2_DIR to <prefix>/lib/cmake/libint2 of a custom build.")
endif()
message(STATUS "Using libint2 ${Libint2_VERSION} from ${Libint2_DIR}")

# Shared libint preferred: builds with high angular momentum are ~1 GB of code,
# which would otherwise be linked into (and reloaded with) every rebuilt module.
option(GEMQC_LIBINT_STATIC "Link libint2 statically (if the installation has it)" OFF)
set(GEMQC_LIBINT_RPATH "")
set(_have_shared FALSE)
if(TARGET Libint2::int-shared OR TARGET Libint2::int-cxx-shared)
  set(_have_shared TRUE)
endif()
if(TARGET Libint2::int-cxx-headeronly-static AND (GEMQC_LIBINT_STATIC OR NOT _have_shared))
  set(GEMQC_LIBINT_TARGET Libint2::int-cxx-headeronly-static)
  message(STATUS "Linking libint2 statically")
else()
  set(GEMQC_LIBINT_TARGET Libint2::cxx)
  # shared: find exactly this libint at run time (not another installation)
  foreach(_t Libint2::int-shared Libint2::int-cxx-shared)
    if(TARGET ${_t})
      get_target_property(_loc ${_t} IMPORTED_LOCATION_RELEASE)
      if(NOT _loc)
        get_target_property(_loc ${_t} IMPORTED_LOCATION)
      endif()
      if(_loc)
        get_filename_component(GEMQC_LIBINT_RPATH "${_loc}" DIRECTORY)
        break()
      endif()
    endif()
  endforeach()
  message(STATUS "Linking libint2 shared (rpath ${GEMQC_LIBINT_RPATH})")
endif()

# two-geminal kernel Operator::delcgtga_delcgtgb (custom libint only)
set(GEMQC_LIBINT_DELCGTGAB FALSE)
find_file(_libint_engine_h engine.h
  PATHS "${Libint2_DIR}/../../../include/libint2" NO_DEFAULT_PATH NO_CACHE)
if(_libint_engine_h)
  file(STRINGS "${_libint_engine_h}" _hit REGEX "delcgtga_delcgtgb")
  if(_hit)
    set(GEMQC_LIBINT_DELCGTGAB TRUE)
  endif()
endif()
message(STATUS "libint2 kernel delcgtga_delcgtgb: ${GEMQC_LIBINT_DELCGTGAB}")
