# gemQC-pilot — quantum chemistry with geminals

Conceived and directed by A. Koehn, University of Stuttgart, 2026; coded using an
AI agent-based approach.

**gemQC-pilot** implements SCF, MP2, MP2-F12 and a pilot version of MP2-F12XG
(`mp2-f12xg-pilot`), as a demonstrator of the F12XG methods.

It's quantum chemistry code for exploring new methods and algorithms: C++ backend
(integrals with [libint](https://github.com/evaleev/libint), Eigen, OpenMP) for the
heavy computations, Python frontend for driving calculations and pilot coding.

Features: basis sets per element or atom; point-group symmetry; RHF, UHF
and ROHF with direct Fock builds and DIIS, quasi-Newton or Newton solvers; stability
analysis; density fitting (df- tasks); dipole/quadrupole moments; MP2/SCS-MP2 energies
(RHF, UHF, ROHF; exact or DF); MP2-F12 with fixed amplitudes (3C, 3*C; CABS singles;
exact or DF); the F12XG pilot (geminals bound to groups of basis functions); orbital
analysis; YAML input, JSON results, HDF5 restart data.

The full documentation is in `docs/` (build it as shown below).

## Installation

Requirements: C++17 compiler, CMake ≥ 3.20, Python ≥ 3.10, libint ≥ 2.7 (with Eigen3
and Boost headers), optionally OpenMP. On macOS with Homebrew:

```bash
brew install cmake ninja libint pybind11 libomp
pip install scikit-build-core pybind11 numpy scipy pyyaml h5py pytest
pip install --no-build-isolation -e .      # editable; C++ rebuilt on import
python -m pytest tests
```

The MP2-F12 and F12XG methods need a custom libint build with higher angular momenta (the
Homebrew libint is limited to l = 4); for that and for Linux see `docs/installation.md`.

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
