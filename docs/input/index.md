# Input

The input is a YAML file with three blocks:

```yaml
systems:
  water:
    coordinates: h2o.xyz     # path relative to the input file
    basis: def2-SVP
    multiplicity: 1
resources:                   # optional
  memory: 4 GB
  threads: 8
tasks:
  - system: water            # the first task must name a system,
    task: oneint             #   later tasks default to the previous one
    print_matrices: true     # further keys are task options
  - task: scf
    convergence: 1e-6
```

Blocks may also be written as lists of one-key maps (`- basis: def2-SVP`). Unknown
keys are errors (with a suggestion for typos); invalid values are reported with the
expected type or choices. Task names without an implementation are skipped.

## Systems

```{include} ../_generated/options_system.md
```

The geometry is given by exactly one of

- `coordinates`: a file relative to the input file, either XYZ (in `units`) or the
  HDF5 (`.h5`) or JSON file of an earlier gemQC run. Stored coordinates are in bohr and
  in the computational frame of that run; `source_system` selects the system if the
  file holds several.
- `geometry`: inline lines `symbol x y z` (in `units`), as a block string or a list.
- `zmatrix`: inline Z-matrix with lines `atom [i r [j angle [k dihedral]]]`, as in
  Molpro or Gaussian. References are line numbers or labels of earlier atoms. Values are
  numbers or (optionally negated) names from `variables`. Distances are in `units` and
  angles in degrees. The first atom is placed at the origin, the second on z and the
  third in the xz plane. Dummy atoms `X`/`Q` are used for the construction only.

```yaml
systems:
  water:
    basis: def2-SVP
    geometry: |
      O  0.0000000  0.0000000  0.3935881
      H  0.7571222  0.0000000 -0.1967941
      H -0.7571222  0.0000000 -0.1967941
  h2o2:
    basis: def2-SVP
    zmatrix: |
      O
      O 1 roo
      H 1 roh 2 a
      H 2 roh 1 a 3 d
    variables: {roo: 1.45, roh: 0.97, a: 100, d: 120}
  again:
    basis: def2-TZVPP
    coordinates: run.h5        # geometry of system 'water' of an earlier run
    source_system: water
```

### Basis sets per element and atom

`basis` and the auxiliary keys `jk_auxbasis`, `c_auxbasis`, `cabs_auxbasis` take a basis
set name (all atoms) or a mapping with the keys

- `default`: all atoms not named otherwise;
- an element symbol, e.g. `H`;
- an atom number (1-based, input order), e.g. `3`;
- an atom label of `geometry`, `zmatrix` or an XYZ file: an entry that differs from the
  element symbol, e.g. `O1` or `H_a` (stored geometries have no labels).

An atom number or label takes precedence over the element, the element over `default`.
The orbital basis must cover all atoms. Auxiliary basis sets may name only some atoms:
the others get the default fitting basis of their own orbital basis (also as task
options, e.g. `c_auxbasis: {O: cc-pVQZ-C}` changes the oxygen fitting functions only).

```yaml
systems:
  water:
    basis: {default: cc-pVTZ, O: aug-cc-pVTZ, H_a: cc-pVQZ}
    jk_auxbasis: {H_a: cc-pVQZ-JK}         # O, H: default of their basis
    symmetry: nosym                        # H_a differs from the other H
    geometry: |
      O     0.0000000  0.0000000  0.3935881
      H_a   0.7571222  0.0000000 -0.1967941
      H    -0.7571222  0.0000000 -0.1967941
```

The output and the result files name such a basis by its canonical name: the most
frequent basis (ties: the first in atom order), then the elements and atoms that
deviate, e.g. `cc-pVTZ;O=aug-cc-pVTZ;4=cc-pVQZ` for methanol (C, O, H, H, H, H) with
`{default: cc-pVTZ, O: aug-cc-pVTZ, 4: cc-pVQZ}`; this form is also accepted as input. Atoms related by
symmetry are equivalent only with the same basis: with `symmetry: auto` a basis that
differs for such atoms lowers the computational group (e.g. C2v to Cs),
an explicitly requested group that the basis sets do not allow is an input error.

Coordinates are converted to bohr. With the default `modify: align` the molecule may be
rotated and translated into the computational frame, but its shape is kept, so
symmetry is only used if the input has it to about 1e-10 bohr; `modify: adapt`
symmetrizes within `symmetry_tolerance` and `modify: off` keeps the input frame, see
[Symmetry](../concepts/symmetry.md). The output lists the geometry in bohr and the
shortest and longest interatomic distances, with a warning if the shortest distance
is below 0.6 Å or above 2.6 Å (a hint at a bohr/Ångström mix-up).

## Resources

```{include} ../_generated/options_resources.md
```

The command line options `-m` and `-t` override these values.

## Tasks

| task | purpose |
|---|---|
| [`oneint`](oneint.md) | one-electron integrals |
| [`scf`, `rhf`, `rohf`, `uhf`](scf.md) | Hartree-Fock |
| [`mp2`](mp2.md) | MP2 and SCS-MP2 energies (RHF, UHF, ROHF) |
| [`mp2-f12`](mp2-f12.md) | MP2-F12 with fixed amplitudes (3C, 3\*C) |
| [`mp2-f12xg-pilot`](mp2-f12xg-pilot.md) | MP2-F12 with geminals bound to basis-function groups (pilot) |
| [`df-...`](df.md) | density-fitted variants of scf, rhf, rohf, uhf, mp2, mp2-f12, mp2-f12xg-pilot |

Dipole and quadrupole moments are computed
by the SCF tasks (option `properties`), see [Properties](properties.md).

```{toctree}
:hidden:

oneint
scf
mp2
mp2-f12
mp2-f12xg-pilot
df
properties
```
