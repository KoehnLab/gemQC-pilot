# Bespoke libint build: decisions

gemQC links a custom libint 2.13.1 build (separate project, generated library installed
under a prefix and found via `-DLibint2_DIR=<prefix>/lib/cmake/libint2`, see
`cmake/Libint.cmake`); the Homebrew libint (max. angular momentum 4, first ERI
derivatives, no one-body derivatives) remains the fallback. This page collects the
decisions for the build; `gemqc._core.libint_info()` prints the limits of the linked
library.

## Current build (`prod-am7b`)

| integral class | derivative order | max. AM per order |
|---|---|---|
| 4-center ERI | 2 | 7, 6, 5 |
| 3-center ERI (pure solid harmonics) | 2 | 8, 7, 6 |
| 2-center ERI (pure solid harmonics) | 2 | 8, 7, 6 |
| one-body (S, T, V, multipoles up to order 4, `opVop`), property derivatives on | 2 | 7, 7, 7 |

- Generator options (libint 2.13.1, CMake): `-DLIBINT2_MAX_AM="7;6;5"
  -DLIBINT2_ENABLE_ERI=2 -DLIBINT2_ERI_MAX_AM="7;6;5" -DLIBINT2_ENABLE_ERI3=2
  -DLIBINT2_ERI3_MAX_AM="8;7;6" -DLIBINT2_ENABLE_ERI2=2 -DLIBINT2_ERI2_MAX_AM="8;7;6"
  -DLIBINT2_ENABLE_ONEBODY=2 -DLIBINT2_ONEBODY_MAX_AM="7;7;7"
  -DLIBINT2_MULTIPOLE_MAX_ORDER=4 -DLIBINT2_ERI3_PURE_SH=ON -DLIBINT2_ERI2_PURE_SH=ON
  -DLIBINT2_DISABLE_ONEBODY_PROPERTY_DERIVS=OFF`; library shared and static
  (`-DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=ON`, `-O3`), about 1.45 GB each.
- `opVop` ($\mathbf p V\cdot\mathbf p$ and $\mathbf p V\times\mathbf p$, 4 components) is
  always generated with the one-body class. Its derivatives need two header-only fixes
  of the C++ `Engine` (number of shell sets with point-charge centers; multi-component
  reconstruction of the charge derivatives), which the build project applies as patches
  and validates by finite differences (L = 0–6).
- Validated in the build project: libint's own tests, finite-difference checks of the
  derivatives at the AM limits (incl. 3-/2-center second derivatives and one-body second
  derivatives at L = 7), and the gemQC test suite.
- Uses in gemQC-pilot: SCF (ERI), density fitting (3-/2-center) and the F12 integrals
  (3-/2-center and pair integrals at high angular momentum, generic two-body operators,
  runtime kernel `delcgtga_delcgtgb` for the F12XG pilot). Derivative integrals are not
  used in this reduced package.
- Cost on a 16 GB Mac (M1, 4 jobs): generator 2.3 h (serial, 1.9 GB tarball), library
  5.4 h (≈2.8 GB per compile job), install 2.8 GB. The tarball is machine-independent;
  on a cluster only the library step is repeated (≈2.5–3 GB per compile job; on Linux
  build position-independent code, e.g. shared and static).

### Linking gemQC

```bash
pip install --no-build-isolation -e . \
  -Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2
```

The module finds the shared library through its rpath (the `lib` directory of this
installation). If the installation is moved or removed, the next automatic rebuild
(`import gemqc` after C++ changes) silently falls back to another libint (e.g.
Homebrew, max. AM 4): rerun the command above with the new prefix and check

```bash
python -c "import gemqc; print(gemqc.libint_info())"
```

(`max_am` 7, all `*_max_deriv` 2; the output header also prints the libint version and
max. l).

The sections below are the original decision list.

## 1. Angular momentum and derivative orders

The generated code grows steeply with angular momentum (AM) and derivative order, so
each integral class gets its own limits (`--with-<class>-max-am=L0,L1,L2` for
derivative orders 0, 1, 2).

| class | use in gemQC | recommendation |
|---|---|---|
| 4-center ERI | SCF, CASSCF; gradients; second derivatives for spin–spin coupling, Hessians | **decide** max AM: 6 (i functions, e.g. cc-pV5Z on first-row atoms) or 7 (cc-pV6Z, or cc-pV5Z with an extra higher-AM function); derivative order 2, with lower AM for derivatives, e.g. `7,6,5` |
| one-body (S, T, V, multipoles) | all; nuclear-attraction derivatives for 1e spin–orbit integrals and gradients | same max AM as ERI; derivative order 2 (1e spin–orbit needs order 1 of V; order 2 for Hessians and some property integrals) |
| 3-center / 2-center ERI | density fitting (RI-J/K, RI-MP2, CD alternatives) planned later | **decide** whether to build them now; if yes, fitting-basis AM one higher than the orbital basis (e.g. 7 or 8), derivative order 1 |
| multipole order | dipole now; quadrupoles, field gradients later | `--with-multipole-max-order=4` |

## 2. Operators

**decide** which operators to include:

- `coulomb` (needed), `nuclear`, `kinetic`, `overlap`, `emultipole` (needed).
- Spin–orbit: the one-electron Breit–Pauli spin–orbit integrals are
  $\langle\mu|\nabla V \times \nabla|\nu\rangle$ and follow from first derivatives of the
  nuclear attraction integrals. The two-electron integrals (for mean-field or full
  spin-other-orbit terms) follow from mixed second derivatives of the ERIs. (Update:
  the one-electron integrals come directly from `opVop`, see above; the two-electron
  ones from the second ERI derivatives.)
- Spin–spin: the dipolar term needs $\langle \mu\nu | (3 r_i r_j - \delta_{ij} r^2)/r^5 |
  \kappa\lambda\rangle$, obtainable from second derivatives of the ERIs (with care for
  the contact term); again the derivative order 2 of the ERIs suffices.
- Scalar relativistic (DKH/X2C): `opVop` ($\mathbf{p} V \cdot \mathbf{p}$ integrals),
  available in recent libint versions; **decide** if needed.
- Range separation (`erf_coulomb`, `erfc_coulomb`) for range-separated methods, and
  Slater/Gaussian geminals (`stg`, `cgtg`) for F12: **decide**. (Update: no rebuild
  needed: `stg`, `stg_x_coulomb`, `cgtg`, `cgtg_x_coulomb` and `delcgtg2` use the
  generated ERI code; Ten-no's Gm tables cover m ≤ 36, enough for AM 7 in all classes.
  Yukawa integrals crash for exponents ~1e-4, so `TwoBodyOperator` requires ζ ≥ 1e-3.)
- Two-geminal kernel `Operator::delcgtga_delcgtgb` (local addition to the custom build,
  needed by F12XG): $(\nabla_1 g_a)\cdot(\nabla_1 g_b)$ for two contracted Gaussian
  geminals, params `std::array<ContractedGaussianGeminal, 2>{a, b}`. CMake detects it in
  the installed `engine.h` (`GEMQC_HAVE_DELCGTGAB`; status line "libint2 kernel
  delcgtga_delcgtgb"); without it the kind `delcgtga_delcgtgb` of `TwoBodyOperator`
  raises at run time (`TwoBodyOperator.has_delcgtgab()`).

## 3. Conventions

- Spherical harmonics: keep the standard libint ordering (m = −l..l); gemQC already
  relies on it for symmetry (`symmetry.ao_operations`) and in `grid.cpp`.
- Cartesian ordering: standard libint ordering (xx, xy, xz, yy, …).
- Normalization: libint normalizes the contracted functions; the basis reader already
  expects this.
- Shell sets ordered "standard" (default); do not switch to the Gaussian/Orca ordering.

## 4. Build and packaging

- Build the generated library once per machine type with the flags used for gemQC
  (`-O3`, `-march` for the cluster nodes, or a portable baseline if nodes differ:
  **decide**); generation for AM 7 with derivatives takes hours and several GB of disk,
  so the result should be archived and reused.
- Shared or static library: static simplifies deployment of the Python module;
  shared saves disk space if several programs use it (**decide**).
- Eigen and Boost (preprocessor) are needed to build libint; gemQC only needs the
  installed headers and the library (`find_package(Libint2 CONFIG)`).
- Record the configure line and the version in the installed `libint2/config.h`;
  gemQC prints the limits (`gemqc._core.libint_info()`) at startup and should check
  them against the basis set (an error with a clear message when an AM or derivative
  order is not available).
- Version: current release (2.13 or later); pin it for reproducibility.

## 5. For the agent setting up the build

1. Generator: configure the libint source with the chosen `--enable-eri=2
   --enable-eri3=... --enable-eri2=... --enable-1body=2 --with-max-am=...
   --with-eri-max-am=...` etc.; build the exported library tarball (`make export`).
2. Build and install the exported library with CMake (`-DCMAKE_INSTALL_PREFIX`,
   `-DLIBINT2_REQUIRE_CXX_API=ON`, compiler and flags as above).
3. Test: libint's own tests; then gemQC with `-DLibint2_DIR=...`: `python -m pytest
   tests` and an SCF with a basis containing the highest AM (e.g. cc-pV6Z on Ne), whose
   energy is compared with another program.
