# Python API

`import gemqc` gives the C++ core (molecules, basis sets, integrals, SCF solvers)
and the Python frontend (input driver, symmetry, guess,
analysis). All quantities are in atomic units; matrices are numpy
arrays.

## Example

```python
import gemqc

mol = gemqc.read_xyz("h2o.xyz")            # coordinates stored in bohr
basis = gemqc.build_basis(mol, "def2-SVP")
mixed = gemqc.build_basis(mol, {"default": "def2-SVP", "O": "def2-TZVPP"})  # per element/atom
S = gemqc.overlap(basis)
T = gemqc.kinetic(basis)
V = gemqc.nuclear_attraction(basis, mol)
X, Y, Z = gemqc.dipole(basis)
XX, XY, XZ, YY, YZ, ZZ = gemqc._core.second_moments(basis, [0.0, 0.0, 0.0])  # about an origin

small = gemqc.BasisSet(mol, {1: [gemqc.ShellDef(0, True, [1.0], [1.0])],   # own basis
                             8: [gemqc.ShellDef(0, True, [5.0], [1.0])]})  # {Z: shells}
# or shells per atom: gemqc.BasisSet(mol, [shells_atom1, shells_atom2, ...])
S12 = gemqc.overlap(small, basis)          # mixed basis <small|basis>

opt = gemqc.ScfOptions()
opt.convergence = 1e-8
wf = gemqc.rhf(mol, basis, opt, gemqc.sad_guess(mol, basis))
wf.energy, wf.C, wf.eps, wf.D

fb = gemqc.FockBuilder(basis)              # direct J/K for any list of densities
(J,), (K,) = fb.compute([wf.D])

mol.charge, mol.multiplicity = 1, 2
wu = gemqc.uhf(mol, basis, opt, [gemqc.sad_guess(mol, basis)])
wu.s2, wu.D_spin, wu.occ_nat, wu.C_nat     # <S^2>, spin density, natural orbitals
wr = gemqc.rohf(mol, basis, opt, [gemqc.sad_guess(mol, basis)])

from gemqc.geminal import f12_operators     # F12: f12 = -exp(-gamma r12)/gamma, ...
ops = f12_operators(gamma=1.0)             # coulomb, f12, f12/r12, f12^2, (grad f12)^2
                                           #   (STG-6G fit; fit="exact": Slater integrals)
V4 = gemqc._core.two_body_4c(ops["f12"], basis)            # (mu nu|f12|la si), small cases
V3 = gemqc._core.two_body_3c(ops["f12"], aux, basis, cabs) # (P|f12|mu x) (naux, n1*n2)
V2 = gemqc._core.two_body_2c(ops["coulomb"], aux)          # (P|Q)
B = gemqc._core.two_body_3c_mo(ops["f12"], aux, basis, Co, C)  # (P|f12|k q), MO (no AO array)
fb12 = gemqc.FockBuilder(basis, 1e-12, ops["f12"])         # J/K and pair_jk with any operator
K = gemqc._core.pair_integrals(ops["f12"], basis, ri, ri, Co, [(0, 1)])  # K(nu, si) = (nu i|f12|si j), nu, si in ri
```

## Driver

```{eval-rst}
.. autofunction:: gemqc.run
```

## Input options

```{eval-rst}
.. automodule:: gemqc.options
   :members: Option, Task, resolve, markdown_table
```

## F12 operators

```{eval-rst}
.. automodule:: gemqc.geminal
   :members:
```

## Analysis

```{eval-rst}
.. automodule:: gemqc.analysis
   :members:
```

## Geometry, basis sets, symmetry, guess, restart

```{eval-rst}
.. automodule:: gemqc.correlation
   :members: frozen_core_orbitals, orbital_spaces, OrbitalSpaces, SpinSpace, spin_fock

.. automodule:: gemqc.transform
   :members: MOIntegrals, ThreeIndexStore

.. automodule:: gemqc.mp2
   :members: mp2_energy, task_mp2

.. automodule:: gemqc.cabs
   :members: ri_basis, cabs_orbitals, RISpace, build_ri_space, cabs_singles

.. automodule:: gemqc.f12
   :members: F12Integrals, ExactF12Integrals, PairIntermediates

.. automodule:: gemqc.mp2_f12
   :members: fixed_amplitude_energies, task_mp2_f12

.. automodule:: gemqc.f12xg
   :members: assign_bf_groups, BfGroups, ExactF12XGIntegrals, task_mp2_f12xg_pilot

.. automodule:: gemqc.geometry
   :members: read_xyz, parse_xyz_text, parse_zmatrix, read_stored, distance_range, atom_labels

.. automodule:: gemqc.basislib
   :members: find_basis_file, load_basis, build_basis, assign_basis, canonical_name,
             basis_names, atom_shells, aux_assignment, jk_auxbasis, c_auxbasis, cabs_auxbasis,
             contraction_pattern

.. automodule:: gemqc.symmetry
   :members: Symmetry, symmetrize_molecule, scf_symmetry, orbital_labels

.. automodule:: gemqc.guess
   :members: sad_guess, atomic_configuration

.. automodule:: gemqc.restart
   :members: save_wavefunction, find_restart, restart_guess
```

## C++ core

```{eval-rst}
.. automodule:: gemqc._core
   :members:
   :undoc-members:
```
