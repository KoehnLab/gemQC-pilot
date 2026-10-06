# Output and files

## Screen output

The run starts with the logo (with version, program name and credits), git revision,
resources and file names, then
prints for each system the symmetry analysis and geometry, and for each task its
results. Progress lines (e.g. SCF iterations) are flushed immediately; lines are at
most 132 characters. Errors are printed on stdout and stderr.

### Orbital character

The `long` and `full` orbital tables show the character
of each orbital. Each orbital is normalized,
$\sum_{\mu\nu} C_{\mu i} S_{\mu\nu} C_{\nu i} = 1$, and the terms
$C_{\mu i} (SC)_{\mu i}$ are summed per atom and angular momentum (Mulliken weights of
the orbital; they may be slightly negative or above 1). The weights are printed by
size, e.g. `0.583 O p  0.398 H s`, until their sum reaches `character` (default 0.6);
long lists continue on the next line. By default symmetry-equivalent atoms are
merged: `H s` if all H atoms are equivalent, else `H(n) s` for the H atoms equivalent
to atom n; `character_atoms: separate` gives `2H s`, `3H s`.

## Results file (JSON)

`<input>.json` contains the program version and git revision, the input path, start
and end time, for each system the geometry (bohr, computational frame), charge,
multiplicity, basis, point group (and that of the nuclei if lowered by basis sets per
atom: `nuclear_point_group`), irreps and the orientation (rotation and origin
relative to the input frame), and for each task the given options and its results
(energies, convergence, orbital energies, labels and occupations, occupation per
irrep, dipole and quadrupole moments, MP2 and MP2-F12 energy components, file names, ...). Tasks without an
implementation have `status: skipped`.

## Restart file (HDF5)

`<input>.h5` holds one group per system and task:

```
/<system>/<task>/            scf, uhf, mp2, mp2_f12,
                             mp2_f12xg
    molecule/                atomic_numbers, coordinates (bohr); attrs charge, multiplicity
    basis/                   attrs name; per element Z<z>: l, pure, nprim, exponents, coefficients
                             several basis sets: Z<z>_<k> per element and basis k, atom_basis (k per atom)
    attrs                    method, energy, converged, iterations, nbf, timestamp, symmetry
    C, eps, occ, D           orbitals (columns), energies, occupations, AO density
    orbital_labels           n.k
    UHF: C_b, eps_b, occ_b, orbital_labels_b, D_spin, C_nat, occ_nat, natural_labels;
         attrs s2, s2_exact
    ROHF: D_spin
    SCF: attrs dipole, quadrupole, properties_origin (centre of mass)
    mp2: attrs energy, correlation_energy, e_os, e_ss, e_singles, scs_energy, e_scf,
         reference, method, frozen, deleted; pair_energies_singlet, _triplet (RHF,
         i >= j) or pair_energies_aa, _bb, _ab (e_ij)
    mp2_f12: e_f12; pair_energies_mp2, pair_energies_3c, pair_energies_3sc
         (no, no, 2: singlet, triplet for i >= j); attrs energy, e_scf, e_reference, e_cabs_singles,
         mp2_correlation_energy, e_mp2_singlet, e_mp2_triplet, e_f12, e_f12_3c,
         e_f12_3sc, e_f12_singlet, e_f12_triplet, correlation_energy, method,
         approximation, cabs_auxbasis, frozen, gamma
    mp2_f12xg: as mp2_f12, with attrs gammas (group exponents) and bf_groups (group 1
         or 2 per basis function) instead of gamma
```

Coordinates and orbitals refer to the computational frame. The basis definition is
stored with the data, so restarts work across basis sets (the orbitals are projected).
