# Density fitting (df- tasks)

Prefixing a task with `df-` runs it with density-fitted (resolution-of-the-identity)
Coulomb and exchange integrals: `df-scf`, `df-rhf`, `df-rohf`, `df-uhf`,
`df-mp2`, `df-mp2-f12`, `df-mp2-f12xg-pilot`. The options are those of the task
without the prefix, plus

```{include} ../_generated/options_df.md
```

```yaml
systems:
  butane:
    coordinates: butane.xyz
    basis: def2-SVP
    jk_auxbasis: def2-universal-JK     # optional; default from lib/basis/auxbasis.yaml
tasks:
  - {system: butane, task: df-scf}
  - {task: scf}                         # exact, starts from the DF orbitals
```

The JK fitting basis is the task option `jk_auxbasis`, else the one used by an earlier
df- task on the system, the system key `jk_auxbasis`, or the default for the orbital
basis from the table `lib/basis/auxbasis.yaml`: an exact entry (e.g. cc-pVDZ:
cc-pVTZ-JK), a basis file `<basis>-JK`, or the first matching pattern (e.g. `def2-*`,
`cc-pV*Z`). aug-cc-pVXZ-JK (X = T, Q, 5; DZ uses TZ) are generated from cc-pVXZ-JK by
`lib/basis/augment_jk.py` and equal Molpro's AVXZ/JKFIT. JK fitting basis
files carry the suffix `-JK` (e.g. `def2-universal-JK.json`). The output names the
fitting basis and how it was chosen; the DF setup is done once per system and shared by
its df- tasks. Fitting basis sets may be given per element or atom, and with basis sets
per atom the default is taken per atom from its orbital basis, see
[Systems](index.md#basis-sets-per-element-and-atom). `df-mp2` uses a correlation fitting basis instead (system key
`c_auxbasis`, files `<basis>-C`, section `c:` of the table), see [mp2](mp2.md); F12
methods also a CABS (`cabs_auxbasis`, files `<basis>-CABS`, section `cabs:`), see
[mp2-f12](mp2-f12.md).

Density fitting uses the Coulomb metric. Its error in the total energy is typically
1e-5 to 1e-4 Eh (e.g. 2.6e-5 Eh for butane RHF/def2-SVP with def2-universal-JK), but
largely cancels in energy differences. With def2-universal-JK (Turbomole's default
`jkbas`), DF-RHF energies agree with Turbomole's RI-JK (`ridft`, `$rik`) to 1e-10 Eh
(for def2-TZVPP also with Molpro's df-hf; Molpro's DEF2-TZVPP-JKFIT
is the same basis). Molpro's DEF2-SVP-JKFIT, however, is a different, smaller fitting
basis. Orbitals, densities and restart data are
shared with the exact tasks, so an exact calculation can start from DF orbitals (and
vice versa).

The three-index integrals are kept in memory as far as they fit into half of the memory
limit (all of them without a limit); the largest blocks are stored and the rest is
recomputed in every Fock build, so that density fitting also works for systems whose
integrals do not fit into memory. The output of the first df- task reports the share of
stored integrals.

The task key `jk_storage` selects how the three-index integrals are kept:

| `jk_storage` | stored | per Fock build |
|---|---|---|
| `auto` (default) | `fitted` if B fits into half of the memory limit (always without a limit), else `raw` | |
| `raw` | integrals (P\|μν), largest blocks as far as they fit, rest recomputed | half transformation, metric step (naux² N n_occ), contraction |
| `fitted` | metric-contracted integrals B = Mᵀ(P\|μν) (V⁻¹ = M Mᵀ), all of them | half transformation, contraction |

The DF setup of a system is shared by its df- tasks; a task whose `jk_storage` (`raw` or
`fitted`) differs from the existing setup builds it anew, `auto` keeps the existing one.

`fitted` avoids the metric step, which dominates the exchange build for large fitting
bases (Co complex, def2-SVP with def2-universal-JK: 4.0 → 0.9 s per UHF Fock build;
butane/def2-TZVPP: 0.13 → 0.045 s), at the cost of storing B for all fitting functions
(naux · N(N+1)/2 values for the shell pairs that survive screening; 293 MB for the Co
complex, 286 MB of raw integrals there). `fitted` fails if B exceeds the memory limit.
Energies agree with `raw` to about 1e-11 Eh.
