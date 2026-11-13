# gemQC — quantum chemistry with geminals

Conceived and directed by A. Koehn, University of Stuttgart, 2026; coded using an
AI agent-based approach.

gemQC is a quantum chemistry code for exploring new methods and algorithms: a C++
backend for the heavy computations (integrals with [libint](https://github.com/evaleev/libint),
Eigen, OpenMP) and a Python frontend that drives calculations and is used for pilot
coding.

Current features:

- YAML input; geometries from XYZ files, inline, as Z-matrix or from earlier runs;
  basis sets in [Basis Set Exchange](https://www.basissetexchange.org) JSON format,
  also per element or atom (orbital and auxiliary basis sets)
- point-group symmetry (largest D2h subgroup)
- RHF, UHF and high-spin ROHF with direct, integral-screened, incremental Fock builds;
  DIIS, quasi-Newton and Newton solvers
- density fitting for all SCF variants, MP2 and MP2-F12 (`df-` tasks)
- properties: dipole and quadrupole moments of SCF densities
- MP2 and SCS-MP2 energies (RHF, UHF, ROHF), exact or density-fitted, with frozen core
- MP2-F12 with fixed amplitudes (RHF; Molpro's 3C and approximations thereof,
  approximation A; STG-nG fit of geminal; CABS singles), exact or density-fitted
- **F12XG pilot**: MP2-F12 with geminals bound to groups of basis functions
- orbital analysis (Mulliken character per orbital)
- results in JSON, restart data (orbitals, densities) in HDF5


```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
input/index
output
concepts/symmetry
concepts/basis
theory
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
developer/index
```
