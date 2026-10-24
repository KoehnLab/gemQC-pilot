# mp2-f12xg-pilot: MP2-F12 with extended geminals

`mp2-f12xg-pilot` (exact integrals) and `df-mp2-f12xg-pilot` (density fitting) are the
pilot of F12XG: MP2-F12 with fixed amplitudes (as
[mp2-f12](mp2-f12.md): ansatz 3, approximations 3C and 3\*C, NOZ, HY1) where the
correlation factor depends on the basis functions of the geminal orbitals,

$$
g_{ij}(1,2) = \sum_{\mu\nu} f_{\mu\nu}(r_{12})\, \chi_\mu(1) \chi_\nu(2)\, C_{\mu i} C_{\nu j},
\qquad f_{\mu\nu} = -\frac{e^{-\gamma_{\mu\nu} r_{12}}}{\gamma_{\mu\nu}},\quad
\gamma_{\mu\nu} = \tfrac12(\tilde\gamma_G + \tilde\gamma_H),
$$

with two basis-function groups $G \ni \mu$, $H \ni \nu$ and one exponent $\tilde\gamma_G$
per group. Each of the three $\gamma_{GH}$ gets its own STG-nG fit (or `geminal: exact`).
RHF; formulas in [Theory](../theory.md).

- `mp2-f12xg-pilot`: exact integrals (small molecules; water/aug-cc-pVTZ: 27 s for the
  pair integrals, 5 s for mp2-f12).
- `df-mp2-f12xg-pilot`: robust density fitting term by term of the group pairs, with the
  fitting basis `c_auxbasis` and the Fock matrices fitted with `jk_auxbasis`, as
  [df-mp2-f12](mp2-f12.md). The DF error is that of df-mp2-f12 for the same exponents
  (water/aug-cc-pVDZ all electrons: 1e-4 Eh); 3-index storage about 1.7 times that
  of df-mp2-f12 (7 instead of 4 large arrays per active orbital).

```{include} ../_generated/options_mp2-f12xg-pilot.md
```

```yaml
systems:
  water: {coordinates: h2o.xyz, basis: cc-pCVDZ-F12}
tasks:
  - {system: water, task: rhf}
  - task: mp2-f12xg-pilot
    gammas: [2.5, 0.9]         # group 1 (core), group 2 (valence), 1/bohr
    bf_groups: core_valence    # or {core_valence: cc-pVDZ-F12}, {atoms: [O]}, {functions: [...]}
    frozen: off
```

## Basis-function groups

- `core_valence`: group 1 holds
  - the **core orbital functions**: $\langle r_A^2\rangle$ of the normalized function about
    its own atom below `core_valence_threshold` (default 1.05) times the HF
    $\langle r^2\rangle$ of the outermost core shell of the element (1s for Li–Ne, 2s/2p
    for Na–Ar, 3s/3p/3d for Ga–Kr; H and He have no core);
  - the **core-correlating functions**: all shells of the orbital basis that are not in
    the valence reference basis, derived per atom from the name (`cc-pCVXZ`,
    `cc-pwCVXZ` → `cc-pVXZ`; also for a [basis per atom](index.md)) or given as
    `{core_valence: <valence basis>}`. If the derived basis is not
    available, only the first criterion is used (noted in the output). Functions of the
    valence basis that the orbital basis recontracts are accepted if they lie in the span
    of the orbital basis on their atom (aug-cc-pwCV5Z of Br: the 4p contraction of
    cc-pV5Z is dropped and 4 more p primitives are uncontracted, which are then the
    core-correlating p functions).

  Group 2 holds all other functions. The radial criterion alone cannot identify the
  core-correlating functions: they lie in the same range of $\langle r^2\rangle$ as
  compact valence functions (C in cc-pCVDZ-F12: extra p at 3.2 times the 1s value, the
  inner 2p function of cc-pVDZ-F12 at 2.8). K–Zn and elements beyond Kr have no
  core reference yet; use the atoms or functions form there.
- `{atoms: [...]}`: group 1 = the listed atoms (1-based numbers or element symbols),
  group 2 = all others.
- `{functions: [...]}`: group 1 or 2 for each basis function (AO order of the output).

Segmented basis sets (def2) split the core orbitals into several functions spread
between core and valence without a gap, so there the core/valence split is a convention;
such sets may not be suited for F12XG.

## Output

- The groups (functions per group and atom, range of $\langle r_A^2\rangle$ for
  `core_valence`), the group exponents and the geminal of each group pair with its fit.
- CABS, CABS singles and the energy table as for [mp2-f12](mp2-f12.md#output).
- JSON: the entries of mp2-f12 (without `gamma`) and `gammas`, `bf_groups` (description),
  `group_sizes`; HDF5 group `mp2_f12xg` (see [Output](../output.md)).

With equal exponents the results equal those of `mp2-f12` (`df-mp2-f12`) for any
groups; for separated fragments with groups by atom they are the sum of `mp2-f12` for
each fragment with its own exponent (both checked in the tests).
