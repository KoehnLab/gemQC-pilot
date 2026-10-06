# Developer guide

For human and agentic developers. The working rules (commits, test examples,
folder boundaries) are in `AGENTS.md`.

## Architecture

```
input.yaml ──parse_input──► InputSpec (systems, resources, tasks)
                             │ options.resolve: checked options with defaults
                             ▼
setup_system ──► System (molecule, basis, symmetry; later wavefunction)
                             │ TASKS[name](system, options, ctx) -> dict
                             ▼
results ──► screen output (Output), <input>.json (write_results),
            <input>.h5 (restart.py)
```

The C++ core (`cpp/`, module `gemqc._core`) does everything that scales: integrals,
Fock builds, SCF solvers, orbital Hessians. The
Python frontend (`python/gemqc/`) parses input, sets up systems, runs tasks, analyses
and stores results, and is the place for pilot implementations of new methods.

| file | content |
|---|---|
| `cpp/src/basis.cpp`, `molecule.cpp`, `elements.cpp` | basis sets (libint shells), molecules |
| `cpp/src/integrals.cpp` | one-electron integrals, also between two basis sets |
| `cpp/src/fock.cpp` | J/K builder interface `JKBuilder`; direct J/K builds (screening, skeleton with symmetry, OpenMP); one-pass J/K of orbital-pair densities (`pair_jk`); any `TwoBodyOperator` (default Coulomb; F12 pair integrals) |
| `cpp/src/df.cpp` | density fitting (`DFBuilder`): metric, hybrid storage of 3-index integrals, J/K, `pair_jk` |
| `cpp/src/scf.cpp`, `scf_detail.h` | RHF/UHF engine (DIIS, QN, Newton) |
| `cpp/src/rohf.cpp` | general open-shell SCF (ROHF) |
| `cpp/src/grid.cpp` | basis function values at points (tests) |
| `cpp/src/hessian.cpp` | orbital Hessian products (Newton solver) |
| `cpp/src/twobody.cpp` | generic two-electron operators (Coulomb, Slater geminal, Yukawa, Gaussian geminals; `TwoBodyOperator`, `compute_quartet`): 2-, 3- and 4-centre AO integrals over mixed basis sets, MO 3-index integrals (`two_body_3c_mo`), pair integrals with occupied indices over the orbital basis (`pair_integrals`) |
| `cpp/bindings/module.cpp` | pybind11 bindings |
| `python/gemqc/driver.py` | `run`, systems, tasks, printout, results file |
| `python/gemqc/options.py` | option tables of all tasks (check, defaults, docs) |
| `python/gemqc/input.py` | YAML parsing |
| `python/gemqc/geometry.py` | geometry input: XYZ, inline, Z-matrix, stored geometries; atom labels |
| `python/gemqc/basislib.py` | basis set library (BSE JSON), basis sets per element/atom (assignment, canonical names), auxiliary basis defaults |
| `python/gemqc/symmetry.py` | point groups (atom types: element and basis), orientation, symmetry-adapted functions |
| `python/gemqc/guess.py`, `restart.py` | SAD guess, HDF5 restart data |
| `python/gemqc/analysis.py` | orbital character |
| `python/gemqc/transform.py` | integral transformation to the correlated orbital spaces (`MOIntegrals`: class `ovov` per spin case, exact via `pair_jk` or 3-index DF, memory or HDF5 scratch file, block access) |
| `python/gemqc/mp2.py` | task mp2: closed-shell and spin-orbital MP2 energy, singles for ROHF, SCS components, pair energies |
| `python/gemqc/geminal.py`, `cabs.py`, `f12.py`, `mp2_f12.py` | F12: geminal (Slater or STG-nG fit) and its operators; RI basis, CABS orbitals, Fock matrices in the RI basis, CABS singles; robust-fitted 3-index F12 integrals or exact pair integrals and pair intermediates V, X, B, C (`f12.F12Integrals`, `ExactF12Integrals`, `PairIntermediates`); task mp2-f12 (energies 3C, 3*C) |
| `python/gemqc/f12xg.py` | F12XG pilot: basis-function groups (core/valence by `<r^2>` and a valence reference basis, atoms, functions), exact pair integrals with group-masked orbitals (`ExactF12XGIntegrals`), task mp2-f12xg-pilot; geminals per group pair in `geminal.XGGeminal` |
| `python/gemqc/correlation.py` | orbital spaces of correlated methods: frozen core table, inner/outer core per element (`core_shells`) and classification of the correlated occupied orbitals by energy rank, frozen/deleted selection per irrep and spin, semicanonical ROHF orbitals (`freeze_first`, `diag_first`) |
| `python/gemqc/properties.py`, `elements.py` | multipole moments (centre of mass); atomic masses |

## J/K builders and density fitting

`JKBuilder` (`fock.h`) is the interface of all J/K builds: `compute`, `pair_jk`,
`set_symmetry` (skeleton build; ignored by DF), `exact` (incremental Fock builds only for
exact integrals), `describe`. `FockBuilder` uses exact 4-index integrals, `DFBuilder`
(`df.h`) density fitting. The SCF functions (`rhf`, `uhf`, `rohf`) and the orbital Hessian take a builder (`jk`, default: exact); the driver
builds one `DFBuilder` per system for df- tasks (`driver.jk_builder`).

`DFBuilder` stores the raw integrals (P|μν) in blocks (fitting shell, orbital shell
pair s1 ≥ s2), ranked by the Schwarz estimate sqrt((P|P)) sqrt((μν|μν)); blocks are
stored in this order as long as they fit into `memory_fraction` (constructor argument,
default 0.5) of the memory limit, the others are recomputed in every build. Change the
default in `df.h` (and the binding in `module.cpp`) if the split turns out unsuitable;
`memory_fraction = 0` gives a fully direct build (used in the tests). The stored blocks
are laid out by fitting shell, then shell pair, so that the blocks of one fitting shell
are contiguous. `half_transform` (Y = (P|μν) A, used by K, `pair_jk`, `mo_three_index`)
assembles per fitting function the lower triangle of (P|μν) as a dense
N × N matrix in cache and multiplies it with A (symmetric product); K is accumulated
per thread from panels of several X^Q = M^T Y (rank updates of the positive and
negative parts of D for symmetric D).

Storage `fitted` (`jk_storage`, constructor argument `storage`; `auto` decides by the
`memory_fraction` budget) stores B^Q = Σ_P M_PQ (P|μν) instead (`build_fitted`, chunks of
shell pairs: raw integrals computed, then M^T R), row-major per Q over the n1 × n2 blocks
of the shell pairs with any unscreened block (`bpairs_`). `fitted_transform` gives
X = M^T Y from B (unpack the lower triangle per Q, symmetric product) or from the raw
integrals; K, `pair_jk` and `mo_three_index` use it, and `fitted_coulomb` gives J from fit
coefficients in either mode. No raw blocks are stored then.

Default auxiliary basis sets: `lib/basis/auxbasis.yaml` (sections `jk`, `c` and `cabs`:
exact names, then a file `<basis>-JK` / `<basis>-C` / `<basis>-CABS`, then patterns);
extend it for new orbital basis sets. Lookup in `basislib.jk_auxbasis`, `c_auxbasis`,
`cabs_auxbasis`; tasks resolve their basis with `driver.aux_basis_name` (task option,
the one used before on the system, system key, table), per atom via
`basislib.aux_assignment`.

Basis sets per element or atom: `basislib.assign_basis` turns a name or a mapping
(`default`, element, atom number, label) into one basis name per atom,
`canonical_name` into the name that identifies the basis everywhere (`BasisSet.name`,
"as before" chains, restart files; `basis_names` reverses it). `build_basis` and
`atom_shells` accept any of these forms; with several basis sets the `BasisSet` is
built from shells per atom. Code that needs the basis of an atom must use
`basis_names(basis.name, zs)`, never `load_basis(basis.name)`.
`driver.setup_system` assigns the basis before the symmetry analysis and passes atom
types (one per distinct basis) to `symmetry.analyze`, which uses them for the
equivalence of atoms (Z for the centre of charge and the inertia axes); equivalent
atoms thus carry identical shells, as `ao_operations` assumes. The RI builder of correlated
methods is `System.df_c` (`mp2.rc_builder`); the CABS and the Fock matrices in the RI
basis are `cabs.RISpace` (`cabs.build_ri_space`).

## Memory

Largest allocations (N basis functions, nmo orbitals, m active orbitals,
np = m(m+1)/2, T threads, n CSFs); the memory limit is `resources: memory` or `-m`.

| part | memory | control |
|---|---|---|
| SCF (RHF/UHF/ROHF) | (6–10 + 4–6 spin/shells) N² + 2·DIIS·N² | DIIS length reduced to fit |
| Fock build | (T+1) · densities · 2 N² (thread accumulators) | threads, then error |
| orbital Hessian | batches of Fock builds | batch size reduced to fit |
| `pair_jk` (orbital-pair integrals) | 2 np N² (result) + T · (f² N m + 2 np f N), f = largest shell | threads reduced to fit, then error |
| density fitting (`DFBuilder`) | stored (P\|μν) blocks ≤ `memory_fraction` · limit; K: 2 naux · N · batch (4 for non-symmetric D) + T · (2 N² + 1024 N + direct blocks of a fitting shell) | largest blocks stored, rest direct; K in batches of factor vectors |
| DF, `jk_storage: fitted` | B: nfit · Σ_pairs n1 n2 (≈ naux N²/2), all stored; K: naux · N · batch (2 for non-symmetric D) + T · (2 N² + 1024 N) | error if B exceeds the limit; `auto` falls back to raw |
| CABS Fock matrices (`cabs.build_ri_space`) | DFBuilder on the RI basis N' = N + N_CABS (stored blocks as above), or FockBuilder | as density fitting |
| DF-MP2-F12 (`f12.F12Integrals`) | (4 N' + 2 o) · naux · o (o active occupied), one pair: ~8 N'² | `ThreeIndexStore`: memory or HDF5 scratch file (`integrals: auto`) |
| MP2-F12 exact (`f12.ExactF12Integrals`) | o(o+1)/2 · (2 N'² + 2 N' o) stored; `pair_integrals`: 2 · pairs · N'² (F), o² · N' N (G, F²) + T · (shell pair · o · N') | as above; batches of pairs by memory |
| full ERI tensor (tests) | N⁴ | error |

## Conventions

- Atomic units internally; conversions only at input and output.
- Double precision; matrices are row-major `Eigen` matrices (`gemqc::Matrix`) in C++
  and numpy arrays in Python; orbitals are columns of `C`.
- AO order within a shell as in libint (Cartesian: CCA order; spherical
  $m = -l, \dots, l$).
- Symmetry: computational group = largest D2h subgroup;
  orbital labels `n.k`; orbitals per irrep in input and output.
- Output: lines of at most 132 characters, compact; flush after progress lines
  (`log::flush()` in C++, `out.flush()` in Python). Errors go to stdout and stderr.
- Restart and results data refer to the computational frame.

## Adding a task

1. Write `task_<name>(system, options, ctx)` in `driver.py` (or a new module): read
   options as `options["key"]` (all keys are present), print with `ctx.out`, return a
   dict of results (scalars and vectors go to the JSON file).
2. Register it in `TASKS` (driver) and add its option table as a `Task` in
   `options.TASKS`; the input check, the defaults and the documentation tables follow
   from there.
3. Store data for later tasks, if useful, in the HDF5 file (`restart.save_orbitals`).
4. Add a page in `docs/input/` that includes `_generated/options_<name>.md`, and tests.

## Adding a C++ function

1. Declare it in `cpp/include/gemqc/<file>.h` with a brief `///` comment (purpose,
   units, ordering); implement it in `cpp/src/`; add new sources to `CMakeLists.txt`.
2. Bind it in `cpp/bindings/module.cpp` with argument names and a docstring.
3. The editable install rebuilds on `import gemqc`.

## libint

gemQC uses a bespoke libint build (angular momentum 7, second ERI and one-body
derivatives, `opVop`); its configuration and the decisions for the next rebuild are in
[libint build](libint_build.md).

```{toctree}
:hidden:

libint_build
```

## Docstrings and comments

- Python: Google style. Every public function has a one-line summary; arguments,
  returns and raised errors (`Args:`, `Returns:`, `Raises:`) where not obvious, with
  units and shapes. Keep them compact.
- C++: brief `///` comments on the public API in the headers (conventions such as
  units, ordering, symmetry); implementation comments where the method is not obvious.

## Tests

`python -m pytest tests`. New methods are validated against independent references:
Molpro energies (`tests/test_reference.py`, inputs in `reference_scf/`), explicit
constructions (e.g. Hessians from full ERIs) and invariances (symmetry on/off).
Reference outputs of other programs are kept in the gitignored folders `reference_*`
(e.g. `reference_f12`); the tests quote the numbers they compare with (Molpro MP2,
MP2-F12, CABS singles; ORCA; Turbomole).

## Documentation

```bash
pip install sphinx myst-parser furo
sphinx-build -b html docs docs/_build/html
```

The option tables are generated from `gemqc.options` at build time
(`docs/_ext/gemqc_options.py`).
