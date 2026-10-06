# Symmetry and orientation

The full point group is detected (printed for information) and its largest subgroup
of D2h is used as the computational group (for Td and D2d: C2v).

Atoms are equivalent only if they have the same element and the same basis set. Basis
sets per atom (see [Systems](../input/index.md#basis-sets-per-element-and-atom)) may
therefore lower the symmetry, e.g. water with one H in another basis is treated in Cs;
the output then gives the point group of the nuclei as well ("lowered from"). The
centre of charge and the inertia axes, which fix the standard orientation, depend on
the nuclei only.

## Orientation

The input frame is kept if its axes already carry the symmetry elements
of the computational group (about the input origin, else about the centre of nuclear
charge), so irrep labels follow the input: e.g. a planar C2v molecule in the xz plane
has its in-plane p orbitals in B1. Otherwise the molecule is reoriented into a
standard orientation (C2 axis along z; planar C2v/D2h molecules in the yz plane; Cs
mirror plane xy).

How far the input geometry may be changed is set by the system key `modify`:

| `modify` | changes | tolerance (bohr) |
|---|---|---|
| `off` | none; symmetry only if the input axes about the input origin carry it | 1e-10 |
| `align` (default) | rotation and translation, distances are kept | 1e-10 |
| `adapt` | also symmetrization (atoms averaged over their images) | `symmetry_tolerance`, default 1e-3 |

With `off` and `align` the geometry is not symmetrized, so e.g. a structure written
with 6 decimals in a general orientation may give a lower group than with `adapt`;
the output prints the largest deviation from exact symmetry.
Coordinates in results, restart and cube files refer to this computational frame;
rotation and origin relative to the input frame are stored in the JSON file.

## Irreps and labels

Irreps are numbered according to the usual XOR scheme (e.g. C2v: 1 A1, 2 B1, 3 B2, 4 A2); the output prints
the numbering with characters and representative functions. Orbitals are labelled
`n.k`, the n-th orbital of irrep k.

## Symmetry in the SCF

The SCF is blocked by irreps (symmetry-adapted functions, separate orthogonalization
and diagonalization per irrep), so orbitals keep exact symmetry; the occupation per
irrep follows aufbau or is fixed with `occupation`. The Fock build computes only
symmetry-unique shell quartets (skeleton Fock matrix, then symmetrized), which saves
up to a factor h (group order) for non-planar and h/2 for planar molecules.
`symmetry: nosym` switches symmetry off; a subgroup (e.g. `c2v`, `cs`) lowers it.
