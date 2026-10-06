# scf, rhf, rohf, uhf: Hartree-Fock

- `scf` and `rhf` (synonyms): RHF for closed-shell singlets, else high-spin ROHF
  with $M-1$ open shells ($M$ = multiplicity).
- `rohf`: the same; a closed-shell singlet gives RHF with a warning.
- `uhf`: unrestricted HF for any multiplicity, $(N+M-1)/2$ alpha electrons.

```{include} ../_generated/options_scf.md
```

## Initial guess and restart

The default `guess: auto` looks for a converged SCF of the same molecule (atoms,
charge, multiplicity and geometry) in the restart file: first the system's own
previous result (e.g. from an earlier run of the same input), else the most recent
one, e.g. a smaller basis earlier in the input. Its orbitals are projected onto the
current basis. Without such data the SAD guess is used; `guess: restart` makes that
an error instead, `restart_from` names the source system (the geometry may then
differ), and `guess: sad` ignores restart data. A UHF restarts from stored alpha and
beta densities (or the total density of an RHF), an ROHF from the closed and open
shell densities. Results are stored under `<system>/scf` (RHF, ROHF) and
`<system>/uhf`.

## Solvers

`diis`
: Roothaan-Hall diagonalization with Pulay DIIS (aufbau occupation).

`qn`
: Quasi-Newton minimization (L-BFGS with a diagonal Hessian preconditioner) on the
  orbital rotations within each irrep; steps that raise the energy are rejected. It
  starts after a few DIIS iterations once the occupation per irrep is stable
  (`qn_start`, `qn_stable`) and keeps that occupation.

`newton`
: Second-order solver with the same switch from DIIS: augmented-Hessian steps with
  the exact orbital Hessian (direct Hessian-vector products, one Fock build per
  micro-iteration) and a trust region. It converges quadratically and handles
  negative curvature; fewer, but more expensive iterations. Not yet available for
  ROHF (falls back to `qn`).

QN and Newton converge to a minimum, whereas DIIS may end on a higher stationary
point, e.g. for stretched bonds.

## Occupation

By default orbitals are occupied by aufbau in each iteration. `occupation` fixes the
number of orbitals per irrep (XOR numbering, see
[Symmetry](../concepts/symmetry.md)): RHF `[3, 1, 1, 0]` (closed shells), ROHF
`{closed: [...], open: [...]}`, UHF `{alpha: [...], beta: [...]}` (either may be
omitted). `freeze_occupation: n` keeps the aufbau occupation after iteration n.

## Orbital printout

`orbitals: short` prints the occupied and 5 virtual orbital energies with their labels
`n.k` (n-th orbital of irrep k); `long` a table up to the third unoccupied orbital,
`full` all orbitals. The tables list the orbital character (see
[Output](../output.md#orbital-character)). UHF prints alpha and beta orbitals in one
table in energetic order with a spin column (a/b), the expectation value $\langle
S^2\rangle$ with its exact value $S(S+1)$, and the natural orbitals of the total
density. ROHF orbitals are ordered closed, open, virtual, each block by energy.
