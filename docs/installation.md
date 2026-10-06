# Installation

## Requirements

- C++17 compiler, CMake ≥ 3.20 (Ninja recommended)
- Python ≥ 3.10 with `numpy`, `pyyaml`, `h5py`; for building `scikit-build-core`, `pybind11`
- libint ≥ 2.7 (C++ API) with its dependencies Eigen3 and Boost (headers); the F12
  methods need a custom build with higher angular momenta (see below)
- OpenMP (optional; on macOS Homebrew's `libomp`)
- `pytest` for the test suite; `sphinx`, `myst-parser`, `furo` for this documentation

On macOS with Homebrew:

```bash
brew install cmake ninja libint pybind11 libomp
pip install scikit-build-core pybind11 numpy scipy pyyaml h5py pytest
```

The Homebrew libint supports angular momenta up to l = 4 (g functions). This suffices
for `scf` and small `mp2` calculations, but not for MP2-F12 and the F12XG pilot (the
auxiliary and CABS basis sets and the pair integrals need higher angular momenta, and
the F12XG pilot uses the kernel `delcgtga_delcgtgb` of the custom build): use a custom
libint build (see [libint build](developer/libint_build.md); gemQC-pilot is tested
with a build for angular momentum 7). Without it, calculations with l above the
limit stop with an error or abort.

On Linux, build libint yourself. Since it is linked into a Python extension module, a
static libint must be compiled position-independent:

```bash
cmake -S <libint-src> -B <libint-build> -DCMAKE_INSTALL_PREFIX=<prefix> \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON     # or -DBUILD_SHARED_LIBS=ON
cmake --build <libint-build> --target install
```

Use the same Eigen3 as for libint via
`-Ccmake.define.Eigen3_DIR=<eigen-prefix>/share/eigen3/cmake`. If no Eigen3
installation is found, its headers are downloaded into the build directory.

## Building

Editable install into the active Python environment; the C++ code is rebuilt
automatically on `import gemqc` after changes (on clusters see
[HPC clusters](#hpc-clusters)):

```bash
pip install --no-build-isolation -e .
```

With a custom libint build (required for the F12 methods, see
[libint build](developer/libint_build.md)):

```bash
pip install --no-build-isolation -e . -Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2
```

A shared libint is preferred and found at run time through the module's rpath (the
library directory of that installation, so no other libint is picked up); a static
libint is used if no shared one is installed, or with
`-Ccmake.define.GEMQC_LIBINT_STATIC=ON`. Builds for high angular momenta are large
(about 1.5 GB for l = 7 with second derivatives), so static linking makes every rebuild
of gemQC slow. The start of each output shows the libint version and its maximum
angular momentum.

Depending on the libint version the config directory may be `<prefix>/cmake/libint2`;
on a system Python use `pip3 install --user`. OpenMP is used if found; disable it with
`-Ccmake.define.GEMQC_OPENMP=OFF`. Build files go to `build/`. For a build outside of pip (development), configure with
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLibint2_DIR=<prefix>/lib/cmake/libint2`,
build with `ninja -C build` and copy `build/_core*.so` into `python/gemqc/`; then run
with `PYTHONPATH=python`. Use `Release`: libint's assertions are active otherwise and
abort on some high angular momentum classes.

## HPC clusters

Installation and the first import (the automatic rebuild of the editable install) can
be very slow if the C++ files are compiled one at a time:

- **Parallel build.** With `--no-build-isolation` the build tools of the environment are
  used; without Ninja, CMake falls back to Makefiles, which build serially. Install
  Ninja and/or set the number of parallel jobs (also before the first import):

  ```bash
  pip install ninja
  export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
  ```

- **Cores.** Login nodes often restrict a user to one or two cores (check `nproc`); build
  in an interactive job instead, e.g. `srun --pty -c 16 bash`.
- **No rebuild on import.** Where gemQC is only run, not developed, switch off the
  automatic rebuild, or use a regular (non-editable) install:

  ```bash
  pip install --no-build-isolation -e . -Ceditable.rebuild=false \
      -Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2
  pip install --no-build-isolation . -Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2
  ```

  Without the automatic rebuild, rerun the same `pip install` command after changes of
  the C++ code (`cpp/`, `CMakeLists.txt`, e.g. after `git pull`); the build is
  incremental, since `build/` is kept. Python changes need no rebuild with the editable
  install, but do with the regular one. `cmake --build build/<tag>` alone is not enough:
  the compiled module is copied into the Python environment on installation.
- **File system.** A build directory on a network file system (NFS, Lustre) is slow; move
  it with `-Cbuild-dir=<local-or-scratch-dir>` (keep it for incremental rebuilds).
- **Diagnostics.** `-Cbuild.verbose=true` shows the compiler calls;
  `python -X importtime -c "import gemqc"` shows where an import spends its time.

## Tests and documentation

The test suite needs the custom libint (F12 tests); `pytest` aborts in the F12 tests with
the Homebrew libint.

```bash
python -m pytest tests
pip install sphinx myst-parser furo
sphinx-build -b html docs docs/_build/html     # open docs/_build/html/index.html
```
