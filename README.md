# gemQC-pilot — quantum chemistry with geminals

Conceived and directed by A. Koehn, University of Stuttgart, 2026; coded using an
AI agent-based approach.

**gemQC-pilot** implements SCF, MP2, MP2-F12 and a pilot version of MP2-F12XG
(`mp2-f12xg-pilot`), as a demonstrator of the F12XG methods.

It is a quantum chemistry code for exploring new methods and algorithms: It features a C++ backend
(integrals with [libint](https://github.com/evaleev/libint), Eigen, OpenMP) for the
heavy computations and a Python frontend for driving calculations and pilot coding. The basis
set library is taken from the [Basis Set Exchange](https://www.basissetexchange.org).
Please cite libint and the Basis Set Exchange, together with the methods used (see
[References](#references)).

Features: basis sets per element or atom; point-group symmetry; RHF, UHF
and ROHF with direct Fock builds and DIIS, quasi-Newton or Newton solvers; 
density fitting (df- tasks); dipole/quadrupole moments; MP2/SCS-MP2 energies
(RHF, UHF, ROHF; exact or DF); MP2-F12 with fixed amplitudes (3C, 3*C; CABS singles;
exact or DF); the F12XG pilot (geminals bound to groups of basis functions); orbital
analysis; YAML input, JSON results, HDF5 restart data.

The full documentation is in `docs/` (build it as shown below).

## Installation

Requirements: C++17 compiler, CMake ≥ 3.20 (Ninja recommended), Python ≥ 3.10, libint ≥ 2.7
(C++ API, with Eigen3 and Boost headers), optionally OpenMP. Details, the build on HPC
clusters and the tests are described in `docs/installation.md`.

The MP2-F12 and F12XG methods need a custom libint build with higher angular momenta
(gemQC-pilot is tested with libint 2.13.1 for l = 7; generator options in
`docs/developer/libint_build.md`). The Homebrew libint is limited to l = 4, which
suffices for SCF and small MP2 calculations only.

### macOS (Homebrew)

```bash
brew install cmake ninja libint pybind11 libomp
pip install scikit-build-core pybind11 numpy scipy pyyaml h5py pytest
pip install --no-build-isolation -e .      # editable; C++ rebuilt on import
```

With a custom libint build, add `-Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2`
to the last command.

### Linux

Build libint yourself. Since it is linked into a Python extension module, a static
libint must be compiled position-independent:

```bash
cmake -S <libint-src> -B <libint-build> -DCMAKE_INSTALL_PREFIX=<prefix> \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON     # or -DBUILD_SHARED_LIBS=ON
cmake --build <libint-build> --target install

pip install scikit-build-core pybind11 ninja numpy scipy pyyaml h5py pytest
pip install --no-build-isolation -e . -Ccmake.define.Libint2_DIR=<prefix>/lib/cmake/libint2
```

Depending on the libint version the config directory may be `<prefix>/lib64/cmake/libint2`
or `<prefix>/cmake/libint2`. Use the same Eigen3 as for libint
(`-Ccmake.define.Eigen3_DIR=<eigen-prefix>/share/eigen3/cmake`). On clusters, build in
parallel (`export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)`) and consider switching off the
automatic rebuild on import (`-Ceditable.rebuild=false`).

### Check

```bash
python -c "import gemqc; print(gemqc.libint_info())"    # max. angular momenta
python -m pytest tests                                  # F12 tests need the custom libint
```

## Example input

```yaml
systems:
  water:
    coordinates: h2o.xyz     # Å, relative to the input file
    basis: def2-SVP
tasks:
  - system: water
    task: scf
    orbitals: long
  - task: mp2
```

```bash
gemqc water.yaml > water.out   # results: water.json, restart data: water.h5
gemqc -a runs/w1 water.yaml    # results: runs/w1.json, restart data: runs/w1.h5
```

## Python

```python
import gemqc

mol = gemqc.read_xyz("h2o.xyz")
basis = gemqc.build_basis(mol, "def2-SVP")
wf = gemqc.rhf(mol, basis, gemqc.ScfOptions(), gemqc.sad_guess(mol, basis))
print(wf.energy)
```

## Documentation

```bash
pip install sphinx myst-parser furo
sphinx-build -b html docs docs/_build/html     # open docs/_build/html/index.html
```

## Layout

```
cmake/          CMake helpers (libint detection)
cpp/            C++ core: include/gemqc/*.h, src/, bindings/ (pybind11 module gemqc._core)
python/gemqc/   Python frontend
lib/basis/      basis set library (BSE JSON)
docs/           documentation (Sphinx, MyST Markdown)
tests/          pytest suite
```

## References

Please cite the software and data used:

- E. F. Valeev, *Libint: a library for the evaluation of molecular integrals of many-body
  operators over Gaussian functions*, version 2.13.1, http://libint.valeyev.net
- B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, J. Chem. Inf.
  Model. **59**, 4814 (2019), [doi:10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725)
  (Basis Set Exchange); please cite also the original papers of the basis sets used, as
  given by the Basis Set Exchange

The closed-shell methods are based on the following works.

**SCF**

- C. C. J. Roothaan, Rev. Mod. Phys. **23**, 69 (1951), [doi:10.1103/revmodphys.23.69](https://doi.org/10.1103/revmodphys.23.69)
- G. G. Hall, Proc. R. Soc. A **205**, 541 (1951), [doi:10.1098/rspa.1951.0048](https://doi.org/10.1098/rspa.1951.0048)
- P. Pulay, Chem. Phys. Lett. **73**, 393 (1980), [doi:10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4) (DIIS)
- P. Pulay, J. Comput. Chem. **3**, 556 (1982), [doi:10.1002/jcc.540030413](https://doi.org/10.1002/jcc.540030413) (DIIS)
- J. Almlöf, K. Faegri, K. Korsell, J. Comput. Chem. **3**, 385 (1982), [doi:10.1002/jcc.540030314](https://doi.org/10.1002/jcc.540030314) (direct SCF, SAD guess)
- M. Häser, R. Ahlrichs, J. Comput. Chem. **10**, 104 (1989), [doi:10.1002/jcc.540100111](https://doi.org/10.1002/jcc.540100111) (integral screening, incremental Fock build)
- M. Dupuis, H. F. King, Int. J. Quantum Chem. **11**, 613 (1977), [doi:10.1002/qua.560110408](https://doi.org/10.1002/qua.560110408) (skeleton Fock matrix)
- J. H. Van Lenthe, R. Zwaans, H. J. J. Van Dam, M. F. Guest, J. Comput. Chem. **27**, 926 (2006), [doi:10.1002/jcc.20393](https://doi.org/10.1002/jcc.20393) (SAD guess)

**Density fitting**

- J. L. Whitten, J. Chem. Phys. **58**, 4496 (1973), [doi:10.1063/1.1679012](https://doi.org/10.1063/1.1679012)
- B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, J. Chem. Phys. **71**, 3396 (1979), [doi:10.1063/1.438728](https://doi.org/10.1063/1.438728)
- O. Vahtras, J. Almlöf, M. W. Feyereisen, Chem. Phys. Lett. **213**, 514 (1993), [doi:10.1016/0009-2614(93)89151-7](https://doi.org/10.1016/0009-2614(93)89151-7)
- F. Weigend, Phys. Chem. Chem. Phys. **4**, 4285 (2002), [doi:10.1039/b204199p](https://doi.org/10.1039/b204199p) (RI-JK, cc-pVXZ-JKFIT)
- F. Weigend, J. Comput. Chem. **29**, 167 (2008), [doi:10.1002/jcc.20702](https://doi.org/10.1002/jcc.20702) (def2-universal-JKFIT)

**MP2 and DF-MP2**

- C. Møller, M. S. Plesset, Phys. Rev. **46**, 618 (1934), [doi:10.1103/physrev.46.618](https://doi.org/10.1103/physrev.46.618)
- S. Grimme, J. Chem. Phys. **118**, 9095 (2003), [doi:10.1063/1.1569242](https://doi.org/10.1063/1.1569242) (SCS-MP2)
- M. Feyereisen, G. Fitzgerald, A. Komornicki, Chem. Phys. Lett. **208**, 359 (1993), [doi:10.1016/0009-2614(93)87156-w](https://doi.org/10.1016/0009-2614(93)87156-w)
- F. Weigend, M. Häser, Theor. Chem. Acc. **97**, 331 (1997), [doi:10.1007/s002140050269](https://doi.org/10.1007/s002140050269)

**MP2-F12 and DF-MP2-F12**

- W. Kutzelnigg, Theor. Chim. Acta **68**, 445 (1985), [doi:10.1007/bf00527669](https://doi.org/10.1007/bf00527669)
- W. Kutzelnigg, W. Klopper, J. Chem. Phys. **94**, 1985 (1991), [doi:10.1063/1.459921](https://doi.org/10.1063/1.459921)
- W. Klopper, C. C. M. Samson, J. Chem. Phys. **116**, 6397 (2002), [doi:10.1063/1.1461814](https://doi.org/10.1063/1.1461814) (auxiliary basis for the RI)
- E. F. Valeev, Chem. Phys. Lett. **395**, 190 (2004), [doi:10.1016/j.cplett.2004.07.061](https://doi.org/10.1016/j.cplett.2004.07.061) (CABS)
- S. Ten-no, Chem. Phys. Lett. **398**, 56 (2004), [doi:10.1016/j.cplett.2004.09.041](https://doi.org/10.1016/j.cplett.2004.09.041) (Slater-type geminal)
- S. Ten-no, J. Chem. Phys. **121**, 117 (2004), [doi:10.1063/1.1757439](https://doi.org/10.1063/1.1757439) (fixed amplitudes)
- D. P. Tew, W. Klopper, J. Chem. Phys. **123**, 074101 (2005) [doi:10.1063/1.1999632](https://doi.org/10.1063/1.1999632) (STG-nG fit)
- F. R. Manby, J. Chem. Phys. **119**, 4607 (2003), [doi:10.1063/1.1594713](https://doi.org/10.1063/1.1594713) (density fitting of the F12 integrals)
- B. I. Dunlap, J. Mol. Struct. (THEOCHEM) **529**, 37 (2000), [doi:10.1016/s0166-1280(00)00528-5](https://doi.org/10.1016/s0166-1280(00)00528-5) (robust fitting)
- S. Kedžuch, M. Milko, J. Noga, Int. J. Quantum Chem. **105**, 929 (2005), [doi:10.1002/qua.20744](https://doi.org/10.1002/qua.20744) (approximation C)
- H.-J. Werner, T. B. Adler, F. R. Manby, J. Chem. Phys. **126**, 164102 (2007), [doi:10.1063/1.2712434](https://doi.org/10.1063/1.2712434) (ansatz 3, 3C, 3\*C, HY1, STG-nG fit)
- T. B. Adler, G. Knizia, H.-J. Werner, J. Chem. Phys. **127**, 221106 (2007), [doi:10.1063/1.2817618](https://doi.org/10.1063/1.2817618) (3C(FIX), CABS singles)
