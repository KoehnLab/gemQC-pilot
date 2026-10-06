"""Input options: one table per task (name, type, default, description).

The tables are the single source for the input check, the defaults used by
the tasks and the input reference of the documentation. Defaults of the SCF
solver options are those of the C++ ScfOptions.
"""

import difflib
from dataclasses import dataclass

from . import _core

KINDS = ("float", "int", "bool", "str", "choice", "list", "map", "basis", "any")


class InputError(ValueError):
    """Error in the user input."""


@dataclass(frozen=True)
class Option:
    """One input option.

    Attributes:
        name: Key in the input.
        kind: Value type, one of KINDS; list accepts a single value as a
            one-element list, basis a name or a mapping per element/atom
            (basislib.assign_basis), any is checked by the task.
        default: Value if the key is not given (None: not set).
        doc: One-line description (Markdown).
        choices: Allowed values of a choice option (lower case).
        unit: Unit of the value, e.g. 'bohr'.
        required: The key must be given.
    """
    name: str
    kind: str
    default: object = None
    doc: str = ""
    choices: tuple = ()
    unit: str = ""
    required: bool = False


@dataclass(frozen=True)
class Task:
    """A task: name, description, options and synonyms (for check and documentation)."""
    name: str
    doc: str
    options: tuple
    synonyms: tuple = ()


def _cpp(name):
    """Default of an SCF solver option from the C++ ScfOptions."""
    return getattr(_core.ScfOptions(), name)


# --- option groups ------------------------------------------------------------------

SOLVER = (
    Option("solver", "choice", _cpp("solver"), "SCF solver: DIIS, quasi-Newton (L-BFGS) "
           "or trust-region Newton (augmented Hessian)", ("diis", "qn", "newton")),
    Option("convergence", "float", _cpp("convergence"), "RMS orbital gradient"),
    Option("energy_convergence", "float", _cpp("energy_convergence"), "energy change",
           unit="Eh"),
    Option("max_iterations", "int", _cpp("max_iterations"), "maximum number of iterations"),
    Option("diis_size", "int", _cpp("diis_size"), "number of DIIS vectors"),
    Option("diis_start", "int", _cpp("diis_start"), "first DIIS iteration"),
    Option("level_shift", "float", _cpp("level_shift"), "level shift of the virtual orbitals",
           unit="Eh"),
    Option("screening", "float", _cpp("screening"), "integral screening threshold"),
    Option("incremental", "bool", _cpp("incremental"), "incremental Fock build"),
    Option("rebuild_every", "int", _cpp("rebuild_every"),
           "full Fock build every n iterations (incremental build)"),
    Option("incremental_until", "float", _cpp("incremental_until"),
           "incremental build until rms(grad) is below this value"),
    Option("lindep", "float", _cpp("lindep"), "cutoff of overlap eigenvalues (linear "
           "dependencies)"),
    Option("qn_start", "int", _cpp("qn_start"), "earliest quasi-Newton/Newton iteration"),
    Option("qn_stable", "int", _cpp("qn_stable"), "quasi-Newton/Newton once the occupation "
           "per irrep was unchanged for this many DIIS iterations"),
    Option("qn_history", "int", _cpp("qn_history"), "number of L-BFGS pairs"),
    Option("qn_max_step", "float", _cpp("qn_max_step"), "maximum rotation angle per "
           "quasi-Newton step", unit="rad"),
    Option("trust_radius", "float", _cpp("trust_radius"), "initial Newton trust radius "
           "(at most twice that)"),
    Option("newton_micro", "int", _cpp("newton_micro"), "maximum micro-iterations per "
           "Newton step"),
)
SOLVER_KEYS = tuple(o.name for o in SOLVER)

GUESS = (
    Option("guess", "choice", "auto", "initial guess: `auto` (restart if possible, else "
           "SAD), `restart`, `sad`, `core`", ("auto", "restart", "sad", "core")),
    Option("guess_basis", "str", "ANO-R0", "minimal basis of the SAD guess"),
    Option("restart_file", "str", None, "HDF5 restart file (default: the one of this run)"),
    Option("restart_from", "str", None, "restart from this system (name) of the restart "
           "file"),
)

SAVE = Option("save", "bool", True, "store the results in the HDF5 file")

PROPERTIES = Option("properties", "bool", True, "dipole and quadrupole moments (origin: centre "
                    "of mass)")

C_AUXBASIS = Option("c_auxbasis", "basis", None, "correlation fitting basis of `df-` "
                    "variants (default: the one used before for this system, else the "
                    "system key, a file `<basis>-C` or the table `lib/basis/auxbasis.yaml`)")

# orbital spaces of correlated methods (gemqc.correlation)
FROZEN = (
    Option("frozen", "any", "auto", "frozen core orbitals: `auto` (noble-gas core per atom, "
           "for Ga-Kr and heavier p-block atoms with the filled d shell), `off`, a number "
           "(the lowest by energy) or a list per irrep"),
    Option("deleted", "any", "off", "deleted virtual orbitals: `off`, a number (the highest by "
           "energy) or a list per irrep"),
    Option("semicanonical", "choice", "freeze_first", "semicanonical orbitals of ROHF "
           "references: `freeze_first` (select frozen/deleted, then diagonalize the active "
           "blocks) or `diag_first` (diagonalize the whole occupied/virtual blocks first)",
           ("freeze_first", "diag_first")),
)

ORBITALS = Option("orbitals", "choice", "short", "orbital printout: `short` (occupied and "
                  "5 virtual, compact), `long` (table up to the third unoccupied orbital) or "
                  "`full` (all)", ("short", "long", "full"))

CHARACTER = (
    Option("character", "float", 0.6, "orbital character in the long/full tables: Mulliken "
           "weights (atom, l) listed until their sum reaches this value; 0: off"),
    Option("character_atoms", "choice", "merged", "symmetry-equivalent atoms `merged` or "
           "`separate`", ("merged", "separate")),
)

HF = SOLVER + GUESS + (
    SAVE,
    Option("occupation", "any", None, "orbitals per irrep: RHF `[3, 1, 1, 0]` or `{closed: "
           "[...]}`, ROHF `{closed: [...], open: [...]}`, UHF `{alpha: [...], beta: [...]}` "
           "(default: aufbau)"),
    Option("freeze_occupation", "int", 0, "keep the aufbau occupation after this iteration "
           "(0: never)"),
    Option("skeleton", "bool", True, "symmetry-unique integrals only in the Fock build"),
    ORBITALS, PROPERTIES,
) + CHARACTER

# --- tasks --------------------------------------------------------------------------

MP2_F12_OPTIONS = FROZEN + (
    Option("approximation", "choice", "3c", "`3c` (coupling to the conventional "
           "amplitudes, F C term in B) or `3*c` (without); both are printed",
           ("3c", "3*c")),
    Option("noz", "bool", False, "omit the Z terms of B (GBC in B; Molpro NOZ)"),
    Option("hy1", "bool", False, "omit the exchange terms of B evaluated by a double RI "
           "expansion (Molpro HY1; Molpro implies NOZ with HY1)"),
    Option("gamma", "float", 1.0, "geminal exponent of f12 = -exp(-gamma r12)/gamma",
           unit="1/bohr"),
    Option("geminal", "any", 6, "number of Gaussians fitted to the Slater geminal "
           "(STG-nG, as Molpro), or `exact` (Slater integrals)"),
    Option("geminal_exponents", "list", None, "explicit Gaussian exponents of the fit "
           "exp(-gamma r) ~ sum c_i exp(-a_i r^2) (with geminal_coefficients)"),
    Option("geminal_coefficients", "list", None, "explicit coefficients c_i of the fit"),
    Option("cabs_singles", "bool", True, "CABS singles correction to the reference "
           "energy"),
    Option("all_electron_cabs", "bool", True, "include the frozen core orbitals in the "
           "CABS singles (off: only the correlated occupied orbitals)"),
    Option("cabs_auxbasis", "basis", None, "complementary auxiliary basis (default: the "
           "one used before for this system, else the system key, a file "
           "`<basis>-CABS` or the table `lib/basis/auxbasis.yaml`)"),
    Option("cabs_threshold", "float", 1e-8, "linear dependencies of the CABS: smallest "
           "eigenvalue of the projected overlap kept"),
    C_AUXBASIS,
    Option("pair_energies", "bool", False, "print the pair energies (always in the JSON and "
           "HDF5 files)"),
    Option("integrals", "choice", "auto", "storage of the transformed integrals: "
           "`memory`, `disk` or `auto` (as for mp2)", ("auto", "memory", "disk")),
    Option("threshold", "float", 1e-12, "integral screening (exact integrals)"),
    SAVE)
# mp2-f12 options not used by mp2-f12xg-pilot (geminal by gammas)
_XG_DROP = ("gamma", "geminal_exponents", "geminal_coefficients")

TASKS = {t.name: t for t in (
    Task("oneint", "One-electron integrals (S, T, V, dipole) and the smallest overlap "
         "eigenvalue.",
         (Option("print_matrices", "bool", False, "print the matrices"),)),
    Task("scf", "Hartree-Fock: RHF for closed-shell singlets, else high-spin ROHF.", HF,
         synonyms=("rhf",)),
    Task("rohf", "High-spin ROHF; RHF (with a warning) for closed-shell singlets.", HF),
    Task("uhf", "Unrestricted Hartree-Fock (any multiplicity).", HF),
    Task("mp2", "MP2 energy on top of the last SCF (RHF; UHF; ROHF in semicanonical "
         "orbitals with singles); SCS-MP2 components, pair energies.",
         FROZEN + (
          Option("pair_energies", "bool", False, "print the pair energies (always in the JSON and "
                 "HDF5 files)"),
          Option("integrals", "choice", "auto", "storage of the transformed integrals (mp2: "
                 "(ia|jb); df-mp2: 3-index B^Q_ia): `memory`, `disk` (HDF5 file in the scratch "
                 "directory) or `auto` (memory if they fit into half of the memory limit)",
                 ("auto", "memory", "disk")),
          Option("threshold", "float", 1e-12, "integral screening (mp2)"),
          C_AUXBASIS,
          SAVE)),
    Task("mp2-f12", "MP2-F12 with fixed amplitudes (ansatz 3, approximations 3C and 3*C) on "
         "top of the last SCF (RHF); exact integrals (small systems) or density fitting.",
         MP2_F12_OPTIONS),
    Task("mp2-f12xg-pilot", "F12XG pilot: MP2-F12 (fixed amplitudes, 3C/3*C) with one "
         "geminal per pair of basis-function groups, gamma_GH = (gt_G + gt_H)/2; exact "
         "integrals (small systems) or density fitting, RHF.",
         tuple(o for o in MP2_F12_OPTIONS if o.name not in _XG_DROP) + (
          Option("gammas", "list", None, "geminal exponents gt_G of the basis-function "
                 "groups (two values)", unit="1/bohr", required=True),
          Option("bf_groups", "any", None, "basis-function groups: `core_valence` (group 1 "
                 "core: core orbital functions and the functions not in the valence basis "
                 "derived from the name, cc-pCVXZ -> cc-pVXZ), `{core_valence: <valence "
                 "basis>}`, `{atoms: [...]}` (group 1: these atoms, numbers or element "
                 "symbols) or `{functions: [...]}` (group 1 or 2 per basis function)",
                 required=True),
          Option("core_valence_threshold", "float", 1.05, "core orbital functions: "
                 "`<r_A^2>` of the basis function about its atom below this multiple of the "
                 "HF `<r^2>` of the outermost core shell"),
         )),
)}
for _t in list(TASKS.values()):
    for _s in _t.synonyms:
        TASKS[_s] = _t

# tasks with a density-fitted variant "df-<task>" (options of the task and DF_OPTIONS)
DF_TASKS = ("scf", "rhf", "rohf", "uhf", "mp2", "mp2-f12", "mp2-f12xg-pilot")
DF_OPTIONS = (
    Option("jk_auxbasis", "basis", None, "JK fitting basis (default: the one used before for "
           "this system, else the system key or the table `lib/basis/auxbasis.yaml`)"),
    Option("jk_storage", "choice", "auto", "3-index storage: `raw` (3-index integrals, largest "
           "blocks stored, rest recomputed), `fitted` (integrals contracted with the metric, "
           "all stored: no metric step in J/K), `auto` (fitted if they fit into half of the "
           "memory limit, else raw)", ("auto", "raw", "fitted")),
)


def task_options(base, df=False):
    """Options of a task: its table, plus DF_OPTIONS for the df- variant."""
    return TASKS[base].options + (DF_OPTIONS if df else ())


def split_task(name):
    """(base task, density fitting) of a task name, e.g. 'df-scf' -> ('scf', True)."""
    n = str(name).lower()
    if n.startswith("df-"):
        return n[3:], True
    return n, False


SYSTEM = (
    Option("coordinates", "str", None, "file (relative to the input file): XYZ, or the "
           "HDF5 or JSON file of an earlier gemQC run (bohr, its computational frame)"),
    Option("source_system", "str", None, "system in an HDF5/JSON `coordinates` file "
           "(default: the only one)"),
    Option("geometry", "str", None, "inline lines `symbol x y z` (instead of a file)"),
    Option("zmatrix", "str", None, "inline Z-matrix, lines `atom [i r [j angle [k "
           "dihedral]]]`, angles in degrees"),
    Option("variables", "map", None, "values of Z-matrix variables"),
    Option("basis", "basis", None, "basis set name, e.g. `def2-SVP`, or per element/atom: "
           "`{default: cc-pVTZ, H: cc-pVDZ, 3: aug-cc-pVTZ, O1: aug-cc-pVTZ}` (atom "
           "number or label > element > default)", required=True),
    Option("jk_auxbasis", "basis", None, "JK fitting basis for `df-` tasks (default: from "
           "`lib/basis/auxbasis.yaml`, see the developer guide; a task option overrides it); "
           "per element/atom as `basis`, atoms not named get the default of their basis"),
    Option("c_auxbasis", "basis", None, "correlation fitting basis for `df-mp2`, "
           "`df-mp2-f12` (default: file `<basis>-C` or `lib/basis/auxbasis.yaml`; a task "
           "option overrides it; per element/atom as `jk_auxbasis`)"),
    Option("cabs_auxbasis", "basis", None, "complementary auxiliary basis (CABS) of F12 "
           "methods (default: file `<basis>-CABS` or `lib/basis/auxbasis.yaml`; a task "
           "option overrides it; per element/atom as `jk_auxbasis`)"),
    Option("units", "choice", "angstrom", "units of the coordinates",
           ("angstrom", "ang", "a", "bohr", "au")),
    Option("charge", "int", 0, "molecular charge"),
    Option("multiplicity", "int", 1, "spin multiplicity 2S + 1"),
    Option("symmetry", "str", "auto", "`auto`, `nosym` or a D2h subgroup, e.g. `c2v`"),
    Option("modify", "choice", "align", "changes of the input geometry: `off` (none; "
           "symmetry only if the input frame carries it), `align` (rotation and "
           "translation), `adapt` (also symmetrization)", ("off", "align", "adapt")),
    Option("symmetry_tolerance", "float", None, "tolerance of the symmetry detection, "
           "only with `modify: adapt` (default 1e-3; `off`, `align`: 1e-10)",
           unit="bohr"),
)

RESOURCES = (
    Option("memory", "str", None, "memory limit, e.g. `4 GB` (bare number: MB; default: "
           "unlimited)"),
    Option("threads", "int", 0, "number of threads (0: all cores)"),
    Option("scratch", "str", None, "directory for scratch files, e.g. on a local SSD "
           "(default: `$GEMQC_SCRATCH`, else the system temporary directory)"),
)


# --- check and defaults -------------------------------------------------------------

def _number(value):
    """YAML 1.1 reads e.g. '1e-6' as a string."""
    return float(value) if isinstance(value, str) else value


def _convert(opt, value, where):
    expected = f"one of {', '.join(opt.choices)}" if opt.choices else f"a {opt.kind} value"
    bad = InputError(f"{where}: invalid value for '{opt.name}': {value!r} (expected "
                     f"{expected})")
    try:
        if opt.kind == "float":
            if isinstance(value, bool):
                raise bad
            return float(_number(value))
        if opt.kind == "int":
            if isinstance(value, bool) or float(_number(value)) != int(float(_number(value))):
                raise bad
            return int(float(_number(value)))
        if opt.kind == "bool":
            if not isinstance(value, bool):
                raise bad
            return value
        if opt.kind == "choice":
            if isinstance(value, bool):  # YAML 1.1 reads off/on/no/yes as bool
                value = {False: "off", True: "on"}[value]
            v = str(value).lower()
            if v not in opt.choices:
                raise bad
            return v
        if opt.kind == "str":
            return str(value)
        if opt.kind == "list":
            return list(value) if isinstance(value, (list, tuple)) else [value]
        if opt.kind == "map":
            if not isinstance(value, dict):
                raise bad
            return dict(value)
        if opt.kind == "basis":
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, (bool, list)):
                raise bad
            return str(value)
    except (TypeError, ValueError):
        raise bad from None
    return value


def check_keys(mapping, allowed, where):
    """Check that all keys of `mapping` are allowed.

    Raises:
        InputError: Unknown key (with a suggestion for typos).
    """
    for k in mapping:
        if k not in allowed:
            hint = difflib.get_close_matches(str(k), list(allowed), n=1)
            msg = f"{where}: unknown key '{k}'"
            raise InputError(msg + (f" (did you mean '{hint[0]}'?)" if hint else ""))


def resolve(options, given, where):
    """Checked values of all options: given ones converted, others the defaults.

    Args:
        options: Tuple of Option.
        given: Mapping of the input (key: value).
        where: Context for error messages, e.g. "task 'scf'".

    Returns:
        dict with every option name (None if neither given nor defaulted).

    Raises:
        InputError: Unknown key, missing required key or invalid value.
    """
    check_keys(given, [o.name for o in options], where)
    out = {}
    for o in options:
        if o.name in given and given[o.name] is not None:
            out[o.name] = _convert(o, given[o.name], where)
        elif o.required:
            raise InputError(f"{where}: '{o.name}' is required")
        else:
            out[o.name] = list(o.default) if isinstance(o.default, list) else o.default
    return out


# --- documentation ------------------------------------------------------------------

def _fmt_default(o):
    if o.required:
        return "required"
    if o.default is None or o.default == []:
        return ""
    if isinstance(o.default, bool):
        return str(o.default).lower()
    return f"{o.default:g}" if isinstance(o.default, float) else str(o.default)


def markdown_table(options):
    """Markdown table of options (key, type, default, description) for the docs."""
    rows = ["| key | type | default | description |", "|---|---|---|---|"]
    for o in options:
        kind = o.kind if o.kind != "choice" else " \\| ".join(o.choices)
        default = _fmt_default(o) + (f" {o.unit}" if o.unit and _fmt_default(o) else "")
        doc = o.doc + (f" ({o.unit})" if o.unit and not _fmt_default(o) else "")
        rows.append(f"| `{o.name}` | {kind} | {default} | {doc} |")
    return "\n".join(rows)
