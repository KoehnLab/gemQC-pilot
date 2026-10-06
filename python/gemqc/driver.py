"""Main program: parse input, set up systems, run tasks."""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import __version__
from . import _core, scratch
from .analysis import Character, wrap_items
from .mp2 import task_mp2
from .f12xg import task_mp2_f12xg_pilot
from .mp2_f12 import task_mp2_f12
from .basislib import (assign_basis, basis_names, build_basis, canonical_name, contraction_pattern,
                       find_basis_file, load_basis, source_name)
from .geometry import (atom_labels, distance_range, parse_xyz_text, parse_zmatrix, read_stored,
                       read_xyz)
from .guess import sad_guess
from .input import InputError, parse_input
from .options import (DF_TASKS, SOLVER_KEYS, TASKS as TASK_OPTIONS, resolve, split_task,
                      task_options)
from .output import Output, bf_labels
from .properties import Multipoles, moment_results, print_moments
from .restart import find_restart, restart_guess, save_wavefunction
from .symmetry import (orbital_irreps, orbital_labels, print_symmetry,
                       scf_symmetry, symmetrize_molecule)
from .units import AU_DIPOLE_IN_DEBYE, length_factor


@dataclass
class System:
    """A molecule with its basis set and the results of earlier tasks.

    Attributes:
        name: System name of the input.
        molecule: Molecule (bohr, computational frame).
        basis: BasisSet.
        symmetry: symmetry.Symmetry.
        wavefunction: Last SCF result (RHF, ROHF, UHF) or None.
        jk_auxbasis: JK fitting basis of the input (name or mapping per element/atom), or
            None (default table).
        c_auxbasis: Correlation fitting basis of the input, or None (default table).
        cabs_auxbasis: CABS of the input, or None (default table).
        labels: Atom labels of the input (geometry, Z-matrix, XYZ file) or None.
        cabs_used: CABS of the last F12 task (default of the next one), or None.
        df_c: RI builder for correlated methods (builder, fitting basis, how chosen) or None.
        df: Density fitting (builder, fitting basis, how chosen) once built, or None.
        multipoles: Multipole integrals about the centre of mass once computed, or None.
    """
    name: str
    molecule: _core.Molecule
    basis: _core.BasisSet
    symmetry: object = None      # symmetry.Symmetry (standard orientation)
    wavefunction: object = None  # last SCF result
    jk_auxbasis: object = None   # JK fitting basis of the input (None: default table)
    c_auxbasis: object = None    # correlation fitting basis of the input (None: default)
    cabs_auxbasis: object = None  # CABS of the input (None: default)
    labels: list = None          # atom labels of the input (basis per atom)
    cabs_used: str = None        # CABS of the last F12 task
    c_used: str = None           # correlation fitting basis of the last correlated task
    df_c: object = None          # RI builder for correlation (fitting basis c_auxbasis)
    scf_jk: object = None        # J/K builder of the last SCF (None: exact integrals)
    df: object = None            # DF builder, built by the first df- task
    multipoles: object = None    # properties.Multipoles, built by the first property run


@dataclass
class RunContext:
    """Printer, file names and results of the current run.

    Attributes:
        out: Output printer.
        h5_path: HDF5 restart file.
        json_path: JSON results file.
        results: [(TaskSpec, results dict or None)].
        started: Start time.
    """
    out: Output
    h5_path: Path    # restart / large data
    json_path: Path  # main results
    results: list = field(default_factory=list)
    started: datetime = field(default_factory=datetime.now)


@lru_cache(maxsize=None)
def git_revision() -> str:
    """SHA of the source repository's HEAD, marked if tracked files are modified."""
    src = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(["git", *args], cwd=src, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    try:
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown (not a git checkout)"
    return sha + (" (with local modifications)" if dirty else "")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_molecule(spec):
    """Molecule of a system spec (XYZ, stored geometry, inline geometry or Z-matrix)."""
    if spec.geometry is not None:
        return parse_xyz_text(spec.geometry, spec.units, spec.name)
    if spec.zmatrix is not None:
        return parse_zmatrix(spec.zmatrix, spec.variables, spec.units, spec.name)
    if not spec.coordinates.exists():
        raise FileNotFoundError(f"coordinates file {spec.coordinates} not found")
    if spec.coordinates.suffix.lower() in (".h5", ".hdf5", ".json"):
        return read_stored(spec.coordinates, spec.source_system, spec.name)[0]
    return read_xyz(spec.coordinates, unit=spec.units, name=spec.name)


def _atom_labels(spec):
    """Atom labels of the input (for basis sets per atom); None for stored geometries."""
    if spec.geometry is not None:
        return atom_labels(spec.geometry)
    if spec.zmatrix is not None:
        return atom_labels(spec.zmatrix, zmatrix=True)
    if spec.coordinates.suffix.lower() in (".h5", ".hdf5", ".json"):
        return None
    return atom_labels(spec.coordinates)


def _check_basis_symmetry(mol, basis, sym):
    """Symmetry-equivalent atoms must carry the same basis set (ensured by the atom
    types of the symmetry analysis; a safeguard for ao_operations).

    Raises:
        ValueError: Equivalent atoms with different basis sets.
    """
    zs = list(mol.atomic_numbers)
    names = basis_names(basis.name, zs)
    for row in sym.atom_map:
        for a, b in enumerate(row):
            if names[a] != names[int(b)]:
                s = mol.symbols
                raise ValueError(f"atoms {a + 1}{s[a]} and {int(b) + 1}{s[int(b)]} are "
                                 f"symmetry-equivalent but have different basis sets")


def _coordinates_source(spec):
    if spec.geometry is not None:
        return f"input (inline, {spec.units})"
    if spec.zmatrix is not None:
        return f"input (Z-matrix, {spec.units}, degrees)"
    if spec.coordinates.suffix.lower() in (".h5", ".hdf5", ".json"):
        src = f", system '{spec.source_system}'" if spec.source_system else ""
        return f"{spec.coordinates.name} (bohr, stored{src})"
    return f"{spec.coordinates.name} ({spec.units})"


def setup_system(spec, out=Output(False)) -> System:
    """Molecule, symmetry and basis of an input system; prints a summary.

    Args:
        spec: input.SystemSpec.
        out: Output printer.

    Returns:
        System.
    """
    try:
        mol = _read_molecule(spec)
    except (ValueError, OSError) as e:
        raise InputError(f"system '{spec.name}': {e}") from None
    mol.charge = spec.charge
    mol.multiplicity = spec.multiplicity
    labels = _atom_labels(spec)
    try:  # basis per atom first: atoms with different basis sets are not equivalent
        names = assign_basis(spec.basis, list(mol.atomic_numbers), labels)
    except ValueError as e:
        raise InputError(f"system '{spec.name}': {e}") from None
    types = [list(dict.fromkeys(names)).index(n) for n in names]
    try:  # reorient into standard orientation of the computational group
        sym = symmetrize_molecule(mol, spec.symmetry_tolerance, spec.symmetry, spec.modify,
                                  types)
    except ValueError as e:
        raise InputError(f"system '{spec.name}': {e}") from None
    try:
        mol.validate()
    except ValueError as e:
        raise InputError(f"system '{spec.name}': {e}") from None
    try:
        basis = build_basis(mol, canonical_name(names, list(mol.atomic_numbers)))
        _check_basis_symmetry(mol, basis, sym)
    except ValueError as e:
        raise InputError(f"system '{spec.name}': {e}") from None
    system = System(spec.name, mol, basis, sym, jk_auxbasis=spec.jk_auxbasis,
                    c_auxbasis=spec.c_auxbasis, cabs_auxbasis=spec.cabs_auxbasis,
                    labels=labels)
    print_system(system, spec, out)
    return system


def print_distances(mol, out):
    """Shortest and longest interatomic distance; warns about implausible values."""
    rng = distance_range(mol)
    if rng is None:
        return
    bohr = length_factor("angstrom")  # bohr per angstrom
    for what, (r, i, j) in zip(("shortest", "longest"), rng):
        pair = f"{i + 1}{mol.symbols[i]}-{j + 1}{mol.symbols[j]}"
        out.value(f"{what} distance", f"{r:12.6f} bohr  {r / bohr:12.6f} Ang   ({pair})")
    rmin = rng[0][0] / bohr
    warn = ("Very short interatomic distances detected!" if rmin < 0.6 else
            "The shortest interatomic distance appears very long!" if rmin > 2.6 else None)
    if warn:
        out(f"  WARNING: {warn} Consider checking your input for Bohr/Angstrom confusion.")
    out()


def _from_note(name, z):
    """Note for elements taken from another basis set file (H, He of p(w)CVXZ sets)."""
    src = source_name(name, z)
    return f"  (from {src})" if src != name else ""


def print_system(system, spec, out):
    """Print the coordinates source, symmetry, geometry and basis of a system."""
    mol, basis = system.molecule, system.basis
    out.section(f"System '{system.name}'")
    out.value("coordinates from", _coordinates_source(spec))
    print_symmetry(system.symmetry, mol.natoms, out)
    out()
    frame = {"off": "input frame", "align": "computational frame",
             "adapt": "computational frame, symmetrized"}[system.symmetry.modify]
    if system.symmetry.group == "C1" and system.symmetry.modify != "off":
        frame = "input frame"
    out(f"  Geometry ({frame}):")
    xyz = mol.coordinates
    rows = [(i + 1, s, *xyz[i]) for i, s in enumerate(mol.symbols)]
    out.table(["#", "atom", "x / bohr", "y / bohr", "z / bohr"], rows,
              ["3d", "4s", "16.10f", "16.10f", "16.10f"])
    out()
    print_distances(mol, out)
    out.value("charge", mol.charge)
    out.value("multiplicity", mol.multiplicity)
    out.value("number of electrons", mol.n_electrons)
    out.value("nuclear repulsion energy", mol.nuclear_repulsion(), "Eh", "18.10f")

    out.section(f"Basis set '{basis.name}'")
    zs = list(mol.atomic_numbers)
    names = basis_names(basis.name, zs)
    if len(set(names)) == 1:
        out.value("file", find_basis_file(basis.name))
        lib = load_basis(basis.name)
        for z in sorted(set(zs)):
            out.value(f"  {_core.element_symbol(z)}",
                      contraction_pattern(lib[z]) + _from_note(basis.name, z))
    else:  # per (element, basis): pattern and atoms
        for b in dict.fromkeys(names):
            out.value(f"file ({b})", find_basis_file(b))
        for z in sorted(set(zs)):
            for b in dict.fromkeys(n for n, x in zip(names, zs) if x == z):
                atoms = ",".join(str(i + 1) for i, (n, x) in enumerate(zip(names, zs))
                                 if x == z and n == b)
                out.value(f"  {_core.element_symbol(z)} {b}",
                          f"{contraction_pattern(load_basis(b)[z])}  atoms {atoms}"
                          + _from_note(b, z))
    out.value("number of shells", basis.nshells)
    out.value("number of basis functions", basis.nbf)
    out.value("max. angular momentum", basis.max_l)


# --- tasks ---
# each task: func(system, options, ctx) -> dict of results; options: all keys of
# the task's table in options.py (given values checked, others the defaults)

def task_oneint(system, options, ctx):
    """One-electron integrals S, T, V, dipole with basic checks.

    Args:
        system: System.
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    out = ctx.out
    mol, basis = system.molecule, system.basis
    S = _core.overlap(basis)
    T = _core.kinetic(basis)
    V = _core.nuclear_attraction(basis, mol)
    D = _core.dipole(basis)

    out.section("One-electron integrals")
    out.value("max |S_ii - 1|", np.abs(np.diag(S) - 1).max(), "", ".2e")
    out.value("max |S - S^T|", np.abs(S - S.T).max(), "", ".2e")
    out.value("smallest eigenvalue of S", np.linalg.eigvalsh(S)[0], "", ".6e")
    out.value("tr T", np.trace(T), "Eh", "18.10f")
    out.value("tr V", np.trace(V), "Eh", "18.10f")
    # nuclear dipole for reference (origin 0)
    nuc = mol.coordinates.T @ np.array(mol.atomic_numbers, dtype=float)
    out.value("nuclear dipole / au", " ".join(f"{x:12.6f}" for x in nuc))
    out.value("nuclear dipole / Debye", " ".join(f"{x:12.6f}" for x in nuc * AU_DIPOLE_IN_DEBYE))

    if options["print_matrices"]:
        labels = bf_labels(mol, basis)
        for title, m in (("Overlap S", S), ("Kinetic T", T), ("Nuclear attraction V", V)):
            out.section(title)
            out.matrix(m, labels)
    return {"S": S, "T": T, "V": V, "dipole": D,
            "smallest_overlap_eigenvalue": float(np.linalg.eigvalsh(S)[0])}


def _scf_options(options):
    """C++ ScfOptions from the (resolved) solver options."""
    opt = _core.ScfOptions()
    for key in SOLVER_KEYS:
        setattr(opt, key, options[key])
    return opt


def _closed_shell(system):
    mol = system.molecule
    return mol.multiplicity == 1 and mol.n_electrons % 2 == 0


def task_scf(system, options, ctx):
    """RHF for closed-shell singlets, else high-spin ROHF (tasks scf, rhf).

    Args:
        system: System.
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    return _hf_task(system, options, ctx, "rhf" if _closed_shell(system) else "rohf")


def task_rohf(system, options, ctx):
    """High-spin ROHF; RHF with a warning for closed-shell singlets.

    Args:
        system: System.
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    if _closed_shell(system):
        return _hf_task(system, options, ctx, "rhf",
                        note="WARNING: ROHF requested for a closed-shell singlet; this is RHF")
    return _hf_task(system, options, ctx, "rohf")


def task_uhf(system, options, ctx):
    """Unrestricted HF (any multiplicity).

    Args:
        system: System.
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    return _hf_task(system, options, ctx, "uhf")


_OCC_KEYS = {"rhf": ("closed",), "rohf": ("closed", "open"), "uhf": ("alpha", "beta")}


def _occupation_input(options, method):
    """Occupation per irrep from the input: RHF list of closed shells (or
    {closed: [...]}), ROHF {closed: [...], open: [...]}, UHF {alpha: [...],
    beta: [...]} (either may be missing). Returns the two lists (or None)."""
    occ = options.get("occupation")
    keys = _OCC_KEYS[method]
    if occ is None:
        return None, None
    if method == "rhf" and not isinstance(occ, dict):
        occ = {"closed": occ}
    if not isinstance(occ, dict) or set(occ) - set(keys):
        form = ", ".join(f"{k}: [...]" for k in keys)
        raise InputError(f"{method}: occupation must be given as {{{form}}}")
    lists = [occ.get(k) for k in keys] + [None]
    return [x if x is None or isinstance(x, list) else [x] for x in lists[:2]]


def aux_basis_name(system, kind, options, used):
    """Auxiliary basis of a task ('jk', 'c' or 'cabs'): the task option <kind>_auxbasis,
    else the one used before for this system (`used`), else the system key or the table.

    Returns:
        (name, how it was chosen).
    """
    from .basislib import aux_assignment
    task = options.get(f"{kind}_auxbasis")
    if not task and used:
        return used, "as before"
    try:  # per atom: task option, system key, table for the orbital basis of the atom
        return aux_assignment(kind, system.basis.name, list(system.molecule.atomic_numbers),
                              system.labels,
                              [(task, f"task option {kind}_auxbasis"),
                               (getattr(system, f"{kind}_auxbasis"), f"input ({kind}_auxbasis)")])
    except ValueError as e:
        raise InputError(f"system '{system.name}': {e}") from None


def jk_builder(system, options, out):
    """J/K builder of a task: None (exact integrals, built where needed) or, for df-
    tasks, the system's DFBuilder (built once; rebuilt if the task asks for another
    fitting basis (jk_auxbasis) or storage (jk_storage; `auto` accepts either))."""
    if not options.get("df"):
        return None
    storage = options.get("jk_storage") or "auto"
    df = system.df
    if df is not None and storage != "auto" and df.builder.fitted != (storage == "fitted"):
        df = None
    if df is not None and options.get("jk_auxbasis") and options["jk_auxbasis"] != df.basis:
        df = None
    if df is None:
        try:
            name, how = aux_basis_name(system, "jk", options, None)
            aux = build_basis(system.molecule, name, auxiliary=True)
        except (ValueError, FileNotFoundError) as e:
            raise InputError(f"system '{system.name}': density fitting: {e}") from None
        system.df = None  # release the old builder first
        t0 = time.time()
        try:
            builder = _core.DFBuilder(system.basis, aux, storage=storage)
        except RuntimeError as e:
            raise InputError(f"system '{system.name}': {e} (use jk_storage: raw)") from None
        system.df = SimpleNamespace(builder=builder, basis=name,
                                    how=how, naux=aux.nbf)
        out(f"  JK fitting basis {name} ({how}), {aux.nbf} functions; setup "
            f"{time.time() - t0:.2f} s")
        out.flush()
    else:
        out(f"  density fitting with {system.df.basis} (as before)")
    return system.df.builder


def _df_results(system, options):
    """JSON entry on the density fitting of a task (empty for exact integrals)."""
    if not options.get("df"):
        return {}
    return {"density_fitting": {"jk_auxbasis": system.df.basis, "naux": system.df.naux,
                                "jk_storage": "fitted" if system.df.builder.fitted else "raw"}}


def _df_c_results(system, df):
    """JSON entry on the correlation fitting of a task (empty for exact integrals)."""
    if not df or system.df_c is None:
        return {}
    return {"density_fitting": {"c_auxbasis": system.df_c.basis, "naux": system.df_c.naux}}


def multipoles(system):
    """Dipole and second-moment integrals of a system (computed once)."""
    if system.multipoles is None:
        system.multipoles = Multipoles(system.molecule, system.basis)
    return system.multipoles


def _scf_properties(system, wf, options, out):
    """Moments of an SCF density (printed); (JSON entries, HDF5 attributes)."""
    if not options["properties"]:
        return {}, {}
    mp = multipoles(system)
    m = mp.moments(np.asarray(wf.D))
    print_moments(mp, m, out)
    res = moment_results(mp, m)
    return res, {"properties_origin": mp.origin, "dipole": m["dipole"],
                 "quadrupole": m["quadrupole"]}


def _hf_task(system, options, ctx, method, note=None):
    """RHF, ROHF or UHF; guess: auto (restart if possible, else SAD), restart, sad,
    core."""
    out = ctx.out
    mol, basis = system.molecule, system.basis
    opt = _scf_options(options)
    task = "uhf" if method == "uhf" else "scf"  # restart group (RHF and ROHF: scf)

    df = "DF-" if options.get("df") else ""
    out.section(df + {"rhf": "SCF (RHF)", "rohf": "SCF (ROHF)", "uhf": "UHF"}[method])
    if note:
        out(f"  {note}")
    jk = jk_builder(system, options, out)
    occ_1, occ_2 = _occupation_input(options, method)
    try:
        sym = scf_symmetry(basis, system.symmetry, occ_1, options["freeze_occupation"],
                           options["skeleton"],
                           occupation_beta=occ_2 if method == "uhf" else None,
                           occupation_open=occ_2 if method == "rohf" else None)
    except ValueError as e:
        raise InputError(f"{task}: {e}") from None
    if method == "rohf" and opt.solver == "newton":
        out("  NOTE: no Newton solver for ROHF yet; quasi-Newton (qn) used instead")
        opt.solver = "qn"
    guess = _scf_guess(system, options, ctx, task, method)
    solve = {"rhf": _core.rhf, "rohf": _core.rohf, "uhf": _core.uhf}[method]
    try:
        if method == "rhf":
            wf = solve(mol, basis, opt, guess[0] if guess else None, sym or _core.ScfSymmetry(),
                       jk=jk)
        else:
            wf = solve(mol, basis, opt, guess or [], sym or _core.ScfSymmetry(), jk=jk)
    except (ValueError, RuntimeError) as e:
        raise InputError(f"{task}: {e}") from None
    system.wavefunction = wf
    system.scf_jk = jk
    labels, per_irrep = print_orbitals(system, wf, out, options["orbitals"],
                                       character=_character(system, options))
    props, attrs = _scf_properties(system, wf, options, out)
    if options["save"]:
        save_wavefunction(ctx.h5_path, system.name, task, wf,
                          symmetry=system.symmetry.group, labels=labels, attrs=attrs)
        out(f"  restart data written to {ctx.h5_path.name} [{system.name}/{task}]")
    res = {"wavefunction": wf, **props, "method": df + wf.method, "energy": wf.energy,
           "e_one": wf.e_one, "e_two": wf.e_two, "e_nuc": wf.e_nuc, "converged": wf.converged,
           "iterations": wf.iterations, "gradient_rms": wf.gradient_rms,
           **_df_results(system, options)}
    if method == "rhf":
        res.update({"orbital_energies": wf.eps, "orbital_labels": labels,
                    "orbital_occupations": np.asarray(wf.occ),
                    "closed_shells_per_irrep": per_irrep})
    elif method == "rohf":
        res.update({"s2": wf.s2, "orbital_energies": wf.eps, "orbital_labels": labels,
                    "orbital_occupations": np.asarray(wf.occ),
                    "closed_shells_per_irrep": per_irrep["closed"],
                    "open_shells_per_irrep": per_irrep["open"]})
    else:
        res.update({"s2": wf.s2, "s2_exact": wf.s2_exact,
                    "orbital_energies_alpha": wf.eps, "orbital_labels_alpha": labels["alpha"],
                    "orbital_occupations_alpha": np.asarray(wf.occ),
                    "orbital_energies_beta": wf.eps_b, "orbital_labels_beta": labels["beta"],
                    "orbital_occupations_beta": np.asarray(wf.occ_b),
                    "natural_occupations": np.asarray(wf.occ_nat),
                    "natural_orbital_labels": labels["natural"],
                    "alpha_orbitals_per_irrep": per_irrep["alpha"],
                    "beta_orbitals_per_irrep": per_irrep["beta"]})
    return res


def _irreps_of(system, wf, C, values, exact):
    """Irrep of each orbital: exact from the blocked SCF, else by projection
    (degenerate sets from equal `values`)."""
    sym = system.symmetry
    if wf.group == sym.group and sym.group != "C1":
        return np.array(exact), 0.0
    return orbital_irreps(C, values, _core.overlap(system.basis), system.basis, sym)


def _character(system, options):
    """Orbital character settings: character (threshold of the accumulated
    weight, 0: off), character_atoms (merged: symmetry-equivalent atoms
    together, or separate)."""
    return Character(system, options["character"], options["character_atoms"] == "merged")


def print_orbitals(system, wf, out, style="short", nvirt=5, character=None, labels=None):
    """Print orbital energies with symmetry labels n.k.

    Args:
        system: System.
        wf: Wavefunction.
        out: Output printer.
        style: 'short' (occupied and nvirt virtual orbitals, compact), 'long' (table
            up to the third unoccupied orbital) or 'full' (all orbitals). UHF: alpha
            and beta in one table with a spin column, plus the natural orbitals.
        nvirt: Virtual orbitals in the short form.
        character: analysis.Character for the long/full tables, or None.
        labels: Labels n.k of the orbitals (default: numbered in the given order).

    Returns:
        (labels, occupied orbitals per irrep); UHF: dicts (alpha, beta, natural);
        ROHF: per irrep a dict of closed and open/active shells.
    """
    sym = system.symmetry
    nirr = len(sym.irreps)
    rohf = wf.method == "ROHF"  # closed and open shells
    given_labels = labels
    sets = [("", wf.C, wf.eps, wf.occ, wf.irreps)]
    if wf.unrestricted:
        sets = [("alpha", wf.C, wf.eps, wf.occ, wf.irreps),
                ("beta", wf.C_b, wf.eps_b, wf.occ_b, wf.irreps_b)]
    labels, per_irrep, dev, rows = {}, {}, 0.0, []
    for spin, C, eps, occ_n, exact in sets:
        occ = np.asarray(occ_n) > 0
        irreps, d = _irreps_of(system, wf, C, eps, exact)
        dev = max(dev, d)
        lab = list(given_labels) if given_labels is not None else orbital_labels(irreps)
        labels[spin], per_irrep[spin] = lab, [int(np.sum(irreps[occ] == k)) for k in range(nirr)]
        if rohf:  # closed and open shells per irrep
            on = np.asarray(occ_n)
            per_irrep = {"closed": [int(np.sum(irreps[on > 1.999] == k)) for k in range(nirr)],
                         "open": [int(np.sum(irreps[(on > 1e-8) & (on < 1.999)] == k))
                                  for k in range(nirr)]}
        what = f"{spin.capitalize()} orbital" if spin else "Orbital"
        if style in ("long", "full"):
            chars = character.items(C) if character and character.enabled else [[]] * len(eps)
            rows += [(float(e), spin[:1], lab[i], int(irreps[i]), float(occ_n[i]), chars[i])
                     for i, e in enumerate(eps)]
        else:
            out()
            out(f"  {what} energies / Eh (label n.k: n-th orbital of irrep k of {sym.group})")
            n_occ = int(occ.sum())
            last_occ = int(np.nonzero(occ)[0].max()) if n_occ else -1
            shown = list(range(min(len(lab), max(n_occ, last_occ + 1) + nvirt)))
            def mark(o):
                return "*" if o > 1.999 or (o > 0 and not rohf) else ("+" if o > 1e-8 else " ")
            for start in range(0, len(shown), 6):
                out("  " + "".join(f"{lab[i]:>7s}{eps[i]:12.6f}" + mark(float(occ_n[i]))
                                   for i in shown[start:start + 6]))
            out("  (* doubly, + singly occupied)" if rohf else "  (* occupied)")
    if rows:
        if wf.unrestricted:  # merge alpha and beta (stable: alpha first at equal energy)
            rows.sort(key=lambda r: r[0])
        out()
        order = None
        if rohf:
            order = "closed, open, virtual"
        _print_orbital_table(rows, sym, out, full=style == "full", spin=wf.unrestricted,
                             order=order, character=character)
    if nirr > 1:
        names = "  ".join(f"{n:>4s}" for n, _ in sym.irreps)
        heads = {"": "closed shells per irrep", "alpha": "alpha orbitals per irrep",
                 "beta": "beta orbitals per irrep", "closed": "closed shells per irrep",
                 "open": "open shells per irrep"}
        out(f"  {'':26s}  {names}")
        for spin in per_irrep:
            out(f"  {heads[spin]:26s}  " + "  ".join(f"{c:4d}" for c in per_irrep[spin]))
    if wf.unrestricted:
        occ_nat = np.asarray(wf.occ_nat)
        irreps, d = _irreps_of(system, wf, wf.C_nat, occ_nat, wf.irreps_nat)
        dev = max(dev, d)
        labels["natural"] = orbital_labels(irreps)
        chars = (character.items(wf.C_nat) if style in ("long", "full") and character
                 and character.enabled else None)
        _print_natural_orbitals(labels["natural"], irreps, occ_nat, sym, out, style, chars,
                                character)
    if dev > 1e-4:
        out(f"  WARNING: orbitals deviate from symmetry (max. {dev:.1e})")
    if rohf:
        return labels[""], per_irrep
    if not wf.unrestricted:
        return labels[""], per_irrep[""]
    return labels, per_irrep


def _shown_rows(occ, full, empty=1e-8, nempty=3):
    """Number of table rows: all (full), else up to the third unoccupied
    orbital (occupation <= empty), at least up to the last occupied one."""
    occ = np.asarray(occ)
    if full or len(occ) == 0:
        return len(occ)
    unocc = np.nonzero(occ <= empty)[0]
    occd = np.nonzero(occ > empty)[0]
    n = unocc[nempty - 1] + 1 if len(unocc) >= nempty else len(occ)
    return int(max(n, occd.max() + 1 if len(occd) else 0))


def _print_natural_orbitals(labels, irreps, occ, sym, out, style, chars=None, character=None,
                            frac=0.01):
    """Natural orbitals of the total density: table (long: up to the third
    with occupation below frac, full: all; with orbital character chars) or
    the fractionally occupied ones (short)."""
    out()
    out(f"  Natural orbitals of the total density (label n.k of {sym.group}), occupation numbers")
    if style in ("long", "full"):
        n = _shown_rows(occ, style == "full", empty=frac)
        if chars:
            out(f"  {character.heading()}")
        out()
        out("        #     n.k  irrep    occupation" + ("  character" if chars else ""))
        for i, (lab, k, x) in enumerate(zip(labels[:n], irreps[:n], occ[:n]), start=1):
            name = sym.irreps[k][0] if k >= 0 else "?"
            head = f"    {i:5d}  {lab:>6s}  {name:<5s}  {x:12.6f}"
            for line in wrap_items(head + "  ", chars[i - 1]) if chars else [head]:
                out(line)
        if n < len(occ):
            out(f"    ... {len(occ) - n} more (orbitals: full prints all)")
        out()
        return
    shown = [i for i, n in enumerate(occ) if frac < n < 2 - frac]
    n2, n0 = int(np.sum(occ >= 2 - frac)), int(np.sum(occ <= frac))
    out(f"  {n2} with occupation > {2 - frac}, {n0} below {frac}, fractional ones:")
    for start in range(0, len(shown), 6):
        out("  " + "".join(f"{labels[i]:>7s}{occ[i]:12.6f} " for i in shown[start:start + 6]))
    if not shown:
        out("  (none)")


def _print_orbital_table(rows, sym, out, full=False, spin=False, order=None, character=None):
    """Orbitals in the given sequence (energetic, or by shells `order` with
    energetic order within each): count, label n.k, (spin a/b,) irrep,
    energy, occupation, character; rows (energy, spin, label, irrep,
    occupation, character items), up to the third unoccupied orbital unless
    full."""
    n = _shown_rows([r[4] for r in rows], full)
    what = "Alpha (a) and beta (b) orbitals" if spin else "Orbitals"
    seq = f"by shells ({order}), each in energetic order" if order else "in energetic order"
    out(f"  {what} {seq} (label n.k: n-th orbital of irrep k of {sym.group})")
    chars = character is not None and character.enabled
    if chars:
        out(f"  {character.heading()}")
    out()
    out("        #     n.k" + ("  spin" if spin else "") +
        "  irrep       energy / Eh    occupation" + ("  character" if chars else ""))
    for i, (e, sp, lab, k, x, items) in enumerate(rows[:n], start=1):
        name = sym.irreps[k][0] if k >= 0 else "?"
        head = (f"    {i:5d}  {lab:>6s}" + (f"  {sp:>4s}" if spin else "") +
                f"  {name:<5s}  {e:16.8f}  {x:12.6f}")
        for line in wrap_items(head + "  ", items) if chars else [head]:
            out(line)
    if n < len(rows):
        out(f"    ... {len(rows) - n} more (orbitals: full prints all)")
    out()


def _scf_guess(system, options, ctx, task="scf", method="rhf", source=None):
    """Initial densities for the SCF: [total] or, restarting UHF from UHF,
    [alpha, beta], ROHF from ROHF [closed, open]; None: core guess inside the
    solver. source (dict, optional) receives the kind of guess.

    auto (default): restart from a stored SCF of the same molecule (same
    geometry) if available in the restart file, else SAD; restart: the same
    but an error if nothing is found; restart_from names the source system
    (geometry may then differ); sad; core."""
    out = ctx.out
    mol, basis = system.molecule, system.basis
    guess = options["guess"]
    path = ctx.h5_path.parent / (options["restart_file"] or ctx.h5_path.name)
    if options["restart_from"] is not None and guess not in ("auto", "restart"):
        raise InputError(f"{task}: 'restart_from' requires guess auto or restart")

    tasks = ("uhf", "scf") if task == "uhf" else ("scf", "uhf")
    if guess in ("auto", "restart"):
        if options["restart_from"] is not None:
            group = find_restart(path, options["restart_from"], tasks, mol, geom_tol=None)
            if group is None or not group.startswith(f"{options['restart_from']}/"):
                group = f"{options['restart_from']}/{task}"
        else:
            group = find_restart(path, system.name, tasks, mol)
        if group is not None:
            try:
                dens = restart_guess(path, group, mol, basis, out, spin=method == "uhf",
                                     shells=method == "rohf")
                if source is not None:
                    source["kind"] = f"restart ({group})"
                return dens
            except (FileNotFoundError, OSError, ValueError) as e:
                if guess == "restart":
                    raise InputError(f"{task}: restart failed: {e}") from None
                out(f"  restart not possible ({e})")
        elif guess == "restart":
            raise InputError(f"{task}: no restart data for this molecule in {path}")
        else:
            out("  no restart data for this molecule found")
    if source is not None:
        source["kind"] = "core Hamiltonian" if guess == "core" else "SAD guess"
    if guess == "core":
        out("  core Hamiltonian guess")
        return None
    return [sad_guess(mol, basis, options["guess_basis"], out)]


TASKS = {
    "oneint": task_oneint,
    "scf": task_scf,
    "rhf": task_scf,
    "rohf": task_rohf,
    "uhf": task_uhf,
    "mp2": task_mp2,
    "mp2-f12": task_mp2_f12,
    "mp2-f12xg-pilot": task_mp2_f12xg_pilot,
}
# --- results file ---

def _jsonable(value):
    """Scalars and vectors for the JSON results file; None if not suitable."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value.tolist()
    if isinstance(value, (list, dict)):  # nested plain data (e.g. stability summary)
        items = value.items() if isinstance(value, dict) else enumerate(value)
        conv = {k: _jsonable(v) for k, v in items}
        if any(v is None for v in conv.values()):
            return None
        return conv if isinstance(value, dict) else list(conv.values())
    return None


def write_results(ctx, spec, systems):
    """Write the JSON results file (program, systems, tasks with options and results)."""
    data = {"program": f"gemQC {__version__}", "git_revision": git_revision(),
            "input": str(spec.path), "started": ctx.started.isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"), "systems": {},
            "tasks": []}
    for name, s in systems.items():
        mol = s.molecule
        data["systems"][name] = {
            "symbols": mol.symbols, "atomic_numbers": mol.atomic_numbers,
            "coordinates_bohr": mol.coordinates.tolist(), "charge": mol.charge,
            "multiplicity": mol.multiplicity, "basis": s.basis.name, "nbf": s.basis.nbf,
            "nuclear_repulsion": mol.nuclear_repulsion(),
            "symmetry": {"point_group": s.symmetry.point_group,
                         **({"nuclear_point_group": s.symmetry.nuclear_point_group}
                            if s.symmetry.nuclear_point_group else {}),
                         "computational_group": s.symmetry.group,
                         "irreps": [n for n, _ in s.symmetry.irreps],
                         "orientation_rotation": s.symmetry.rotation.tolist(),
                         "orientation_origin_bohr": s.symmetry.origin.tolist(),
                         "modify": s.symmetry.modify,
                         "tolerance_bohr": s.symmetry.tolerance,
                         "max_deviation_bohr": s.symmetry.max_displacement}}
    for task, res in ctx.results:
        entry = {"task": task.task, "system": task.system, "options": task.options}
        if res is None:
            entry["status"] = "skipped"
        else:
            entry["results"] = {k: v for k, v in
                                ((k, _jsonable(v)) for k, v in res.items()) if v is not None}
        data["tasks"].append(entry)
    ctx.json_path.write_text(json.dumps(data, indent=2))


def error_message(e):
    """One-line error message; the exception type is named unless it is an input error."""
    kind = "" if isinstance(e, (InputError, FileNotFoundError)) else f"{type(e).__name__}: "
    return f"gemqc: error: {kind}{e}"


def run(input_path, quiet=False, memory=None, threads=None, write_files=True, archive=None):
    """Run all tasks of an input file.

    Args:
        input_path: YAML input file.
        quiet: No screen output.
        memory: Memory limit in bytes (overrides the input).
        threads: Number of threads (overrides the input).
        write_files: Write the JSON file (and by default the HDF5 restart data).
        archive: Path and base name of the results (.json), restart (.h5) and cube
            files (default: those of the input file); a suffix .json or .h5 is ignored
            and missing directories are created.

    Returns:
        [(TaskSpec, results dict)] in input order.

    Raises:
        InputError: Invalid input (also printed on stdout).
    """
    out = Output(not quiet)
    # C++ output through the Python printer (keeps order), or silenced
    _core.set_output(out)
    _core.set_output_enabled(not quiet)
    t0 = time.perf_counter()
    status = "terminates with error"
    try:
        results = _run(Path(input_path), out, memory, threads, write_files, archive)
        status = "terminates"
        return results
    except Exception as e:
        # errors also on stdout (the caller reports them on stderr): batch jobs often split both
        out()
        out(error_message(e))
        e.gemqc_reported = True
        raise
    finally:
        out(f"gemQC run {status} at {_now()}  (wall time {time.perf_counter() - t0:.2f} s)")
        _core.set_output(None)
        _core.set_output_enabled(True)


def _run(input_path, out, memory, threads, write_files, archive=None):
    li = _core.libint_info()
    out.logo(__version__)
    out(f"gemQC run starts at {_now()}")
    out()
    out.value("git revision", git_revision())
    out(f"  libint {li['version_major']}.{li['version_minor']}.{li['version_micro']}"
        f" (max l = {li['max_am']}, fitting basis {li['max_am_auxiliary']})")
    spec = parse_input(input_path)
    resolved = []
    for t in spec.tasks:
        base, df = split_task(t.task)
        if df and base not in DF_TASKS:
            raise InputError(f"task '{t.task}': no density-fitted variant (available: "
                             + ", ".join("df-" + x for x in DF_TASKS) + ")")
        resolved.append(resolve(task_options(base, df), t.options, f"task '{t.task}'")
                        if base in TASK_OPTIONS else t.options)

    _core.set_memory(memory if memory is not None else spec.resources.memory)
    _core.set_threads(threads if threads is not None else spec.resources.threads)
    scratch.set_directory(spec.resources.scratch)
    stem = spec.path.with_suffix("")
    if archive is not None:
        stem = Path(archive).expanduser().resolve()
        if stem.suffix.lower() in (".json", ".h5"):
            stem = stem.with_suffix("")
        if write_files:
            stem.parent.mkdir(parents=True, exist_ok=True)
    # append (not with_suffix): base names may contain dots, e.g. job.296471-20260928
    ctx = RunContext(out, stem.parent / (stem.name + ".h5"), stem.parent / (stem.name + ".json"))
    out.value("input", spec.path)
    out.value("threads", _core.threads())
    out.value("memory limit", _core.format_bytes(_core.memory()) if _core.memory() else "none")
    out.value("scratch directory", scratch.directory())
    if write_files:
        where = "" if ctx.h5_path.parent == spec.path.parent else f"   (in {ctx.h5_path.parent})"
        out.value("results / restart files", f"{ctx.json_path.name}, {ctx.h5_path.name}{where}")

    systems = {name: setup_system(s, out) for name, s in spec.systems.items()}

    for n, (task, options) in enumerate(zip(spec.tasks, resolved), start=1):
        base, df = split_task(task.task)
        func = TASKS.get(base)
        if df:
            options["df"] = True
        out.section(f"Task {n}: {task.task} (system '{task.system}')")
        if func is None:
            out("  not implemented yet -- skipped")
            ctx.results.append((task, None))
            continue
        if not write_files and base == "scf" and "save" not in task.options:
            options["save"] = False
        ctx.results.append((task, func(systems[task.system], options, ctx)))
    if write_files:
        write_results(ctx, spec, systems)
    out()
    return ctx.results

