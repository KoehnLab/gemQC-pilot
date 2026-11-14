cmake_minimum_required(VERSION 3.20)
project(gemQC VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- dependencies ---
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
include(Libint)   # provides GEMQC_LIBINT_TARGET, GEMQC_LIBINT_RPATH

# OpenMP (optional); Apple clang needs Homebrew's libomp
option(GEMQC_OPENMP "Enable OpenMP parallelization" ON)
if(GEMQC_OPENMP)
  if(APPLE AND NOT OpenMP_ROOT)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE _omp_prefix
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_omp_prefix)
      set(OpenMP_ROOT "${_omp_prefix}")
    endif()
  endif()
  find_package(OpenMP COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found, building without (set -DGEMQC_OPENMP=OFF to silence)")
  endif()
endif()

# --- C++ core library (linked into the Python module) ---
add_library(gemqc_core STATIC
  cpp/src/basis.cpp
  cpp/src/df.cpp
  cpp/src/elements.cpp
  cpp/src/fock.cpp
  cpp/src/grid.cpp
  cpp/src/hessian.cpp
  cpp/src/info.cpp
  cpp/src/integrals.cpp
  cpp/src/log.cpp
  cpp/src/molecule.cpp
  cpp/src/resources.cpp
  cpp/src/rohf.cpp
  cpp/src/scf.cpp
  cpp/src/twobody.cpp
)
target_include_directories(gemqc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
target_link_libraries(gemqc_core PUBLIC ${GEMQC_LIBINT_TARGET})
if(GEMQC_LIBINT_DELCGTGAB)
  target_compile_definitions(gemqc_core PUBLIC GEMQC_HAVE_DELCGTGAB)
endif()
set_target_properties(gemqc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gemqc_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(gemqc_core PUBLIC GEMQC_HAVE_OPENMP)
endif()

# --- Python extension module gemqc._core ---
pybind11_add_module(_core cpp/bindings/module.cpp)
target_link_libraries(_core PRIVATE gemqc_core)
# keep build rpath on install (avoids install_name_tool rewrites on macOS)
set_target_properties(_core PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
if(GEMQC_LIBINT_RPATH)
  set_target_properties(_core PROPERTIES INSTALL_RPATH "${GEMQC_LIBINT_RPATH}")
endif()
install(TARGETS _core LIBRARY DESTINATION gemqc)
# basis set library in the package (regular install; the editable one uses lib/basis)
if(NOT SKBUILD_STATE STREQUAL "editable")
  install(DIRECTORY lib/basis/ DESTINATION gemqc/basis
          FILES_MATCHING PATTERN "*.json" PATTERN "*.yaml" PATTERN "README.md")
endif()
