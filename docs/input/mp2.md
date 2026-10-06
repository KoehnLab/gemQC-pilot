# mp2: second-order Møller–Plesset energy

`mp2` computes the MP2 correlation energy on top of the last SCF of the system; `df-mp2`
uses density-fitted integrals with a correlation fitting basis. The equations are
the spin-orbital MP2 equations:

- **RHF**: the closed-shell form.
- **UHF**: in the UHF orbitals (same-spin αα, ββ and opposite-spin αβ pairs); the output
  lists ⟨S²⟩ of the reference.
- **ROHF**: in semicanonical orbitals (the occupied and virtual blocks of the α and β
  Fock matrices diagonalized), plus the singles term
  $\sum_{\sigma ia} |f^\sigma_{ia}|^2/(\varepsilon_i - \varepsilon_a)$, since these orbitals
  do not satisfy the Brillouin condition (ROHF-MBPT(2), RMP2 of Knowles et al.). The spin
  Fock matrices are built with the integrals of the SCF (exact, or the JK fitting basis
  after `df-rohf`).

**SCF convergence.** Since the MP2 energy is not stationary with respect to the orbitals,
its error is of the order of the SCF residual (rms orbital gradient, `convergence` of
the SCF task), not of its square as for the SCF energy. The default 1e-6 is fine for
practical computations, but differences of about 1e-6 Eh to other programs can come from
the SCF convergence of either side; use e.g. `convergence: 1e-9` for comparisons. Example
(reference_mp2/cu-nh3-*: Cu(NH3)+, aug-cc-pVTZ, DF-RHF/DF-MP2, Cu JK fitting basis
def2-universal-JK): with the default the MP2 correlation energy differs from Molpro by
1.2 µEh; with `convergence: 1e-9` in gemQC and `gthresh,energy=1d-8,orbital=1d-8` in
Molpro the two agree to 2 nEh, while the SCF energies agree to 2e-10 Eh already with the
default settings.

```{include} ../_generated/options_mp2.md
```

```yaml
  - {system: water, task: scf}
  - task: df-mp2
    frozen: auto              # default; off, a number, or a list per irrep
    deleted: off              # a number (highest virtuals) or a list per irrep
    pair_energies: true
```

## Orbital spaces

- **Frozen core** (`frozen: auto`): the noble-gas core of the previous row of each
  atom (C–Ne: 1s; Na–Ar: [He]2s2p; K–Zn: [Ne]3s3p), for Ga–Kr and the heavier p-block
  atoms including the filled d shell (Ga–Kr: [Ar]3d, 14 orbitals; In–Xe: [Kr]4d). A
  number freezes the lowest occupied orbitals by energy, a list gives the count per
  irrep.
- **Output of the spaces**: frozen, correlated occupied, virtual and deleted orbitals per
  irrep and in total. The correlated occupied orbitals are classified as 
  *inner core*, *outer core* or *valence* (core rows only if not empty): the core of
  each atom is that of `frozen: auto`, its outer core the core subshells of the highest
  $n$ (e.g. Cu: inner core 1s2s2p, outer core 3s3p; O: outer core 1s), and all occupied
  orbitals (frozen included) are assigned by orbital energy: the lowest are inner core,
  then outer core, then valence. Example (Cu(NH3)+, `frozen: off`):

  ```
    orbitals per irrep                    A'   A''   total
      frozen                               0     0       0
      occupied (inner core)                4     1       5
      occupied (outer core)                4     1       5
      occupied (valence)                   6     3       9
      virtual                            114    75     189
      deleted                              0     0       0
  ```
- **Deleted virtuals** (`deleted`): the highest virtual orbitals by energy (a number) or
  per irrep (a list).
- **Semicanonical orbitals** (`semicanonical`, ROHF only): `freeze_first` (default)
  selects the frozen and deleted orbitals in the ROHF orbitals and diagonalizes the
  active blocks; `diag_first` diagonalizes the whole occupied and virtual blocks first
  and selects frozen and deleted orbitals among the semicanonical orbitals (per spin).
  The energies differ slightly.

## Integrals

- `mp2`: (ia|jb) from the AO exchange matrices of the occupied orbital pairs (one pass
  over the shell quartets per batch of pairs); memory $o^2v^2$ per spin case.
- `df-mp2`: $(ia|jb) \approx \sum_Q B^Q_{ia}B^Q_{jb}$ with the fitted 3-index quantities
  (memory $o\,v\,n_{\text{fit}}$ per spin); much faster (butane/def2-TZVPP: 0.4 s instead
  of 15 s), error about 1e-4 Eh.
- `integrals: auto | memory | disk`: the stored arrays are kept in memory if they fit
  into half of the memory limit, else in an HDF5 file in the scratch directory
  (`resources: scratch`), read in blocks and deleted at the end.
- The correlation fitting basis is the task option `c_auxbasis`, else the one of an
  earlier task on the system, the system key `c_auxbasis`, a file `<basis>-C` (e.g.
  `def2-SVP-C`, `def2-TZVPP-C`, `cc-pVTZ-C`, `aug-cc-pVTZ-C`), or the section `c:` of
  `lib/basis/auxbasis.yaml`; per element or atom as described in
  [Systems](index.md#basis-sets-per-element-and-atom). The SCF before `df-mp2` may be exact or `df-scf` (with the
  JK fitting basis).

## Output

The correlation energy is split into opposite-spin (OS) and same-spin (SS) parts (and
singles for ROHF); SCS-MP2 (Grimme) uses $\tfrac65 E_{OS} + \tfrac13 E_{SS}$. If core
orbitals are correlated (inner or outer core, e.g. `frozen: off`), the correlation energy
is also split by pairs into *core* (both orbitals core), *core-valence* (one) and
*valence* (none) correlation (ROHF: plus the singles), printed and in the results as
`e_core`, `e_core_valence`, `e_valence`; in the pair energies the core orbitals are then
marked with `*`. With
`pair_energies: true` the pair energies of the correlated occupied orbitals (orbital labels
$n.k$ as in the SCF output, pairs in energy order) are printed: for closed shells the
singlet and triplet parts of the pair $ij$ ($i \ge j$, $ij$ and $ji$ together; no triplet
for $i = j$), for open shells $e_{ij}$ per spin case ($e_{ij} + e_{ji}$ for same spin).
The results contain `energy` (MP2 total), `correlation_energy`, `e_os`, `e_ss`,
`e_singles`, `scs_energy`, the orbital spaces per irrep (`frozen_per_irrep`,
`occupied_per_irrep` with `inner_core`, `outer_core`, `valence`, `virtual_per_irrep`,
`deleted_per_irrep`; alpha spin), always the pair
energies (`pair_energies`: lists `i`, `j` (labels $n.k$), `singlet`, `triplet` (0 for
$i = j$); open shells per spin case `alpha-alpha`, `beta-beta`, `alpha-beta` with lists
`i`, `j`, `e`); the HDF5 group `<system>/mp2` holds the energies (attributes) and the
pair energies.
