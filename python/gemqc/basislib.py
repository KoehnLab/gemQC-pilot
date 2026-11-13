"""Basis set library: locate and read basis sets in BSE JSON format; basis sets per atom.

Search path: $GEMQC_DIR/lib/basis, then lib/basis of the source tree (editable install),
then the copy installed with the package (regular install).

A basis is given by a name (all atoms) or per atom: a mapping with the keys `default`,
element symbols, atom numbers (1-based) and atom labels (precedence: atom > element >
default), or its canonical name "default;El=name;n=name" (most frequent basis first,
then elements and atoms that deviate). The canonical name identifies a basis everywhere
(BasisSet.name, output, restart files).
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from ._core import BasisSet, ShellDef, element_symbol, element_z

_SRC_BASIS_DIR = Path(__file__).resolve().parents[2] / "lib" / "basis"
_PKG_BASIS_DIR = Path(__file__).resolve().parent / "basis"
_LNAMES = "spdfghiklmnoqrtuvwxyz"  # angular momentum letters


def search_path():
    """Directories searched for basis set files: $GEMQC_DIR/lib/basis, lib/basis, package."""
    dirs = []
    if "GEMQC_DIR" in os.environ:
        dirs.append(Path(os.environ["GEMQC_DIR"]) / "lib" / "basis")
    dirs += [_SRC_BASIS_DIR, _PKG_BASIS_DIR]
    return [d for d in dirs if d.is_dir()]


def _file_stem(name: str) -> str:
    """BSE file naming: lower case, '*' -> '_st_', '+' -> '_pl_', '/' -> '_sl_'."""
    return (name.lower().replace("*", "_st_").replace("+", "_pl_")
            .replace("/", "_sl_"))


def find_basis_file(name: str) -> Path:
    """JSON file of basis set `name` (case-insensitive, BSE file naming).

    Raises:
        FileNotFoundError: Not found in the search path.
    """
    wanted = {name.lower(), _file_stem(name)}
    for d in search_path():
        for f in sorted(d.glob("*.json")):
            if f.stem.lower() in wanted:
                return f
    raise FileNotFoundError(
        f"basis set '{name}' not found in: "
        + ", ".join(str(d) for d in search_path()))


@lru_cache(maxsize=None)
def _read_json(name: str) -> dict:
    return json.loads(find_basis_file(name).read_text())


def ecp_elements(name: str) -> set:
    """Nuclear charges for which basis `name` uses an effective core potential."""
    return {int(z) for z, el in _read_json(name)["elements"].items()
            if "ecp_potentials" in el}


_CORE_NAME = re.compile(r"pw?CV(?=[DTQ1-9]Z)", re.IGNORECASE)


def core_fallback(name: str, z: int):
    """Name of the basis used for H and He with a core-valence basis: any p(w)CVXZ set
    (all types, e.g. aug-cc-pwCVTZ-C) falls back to pVXZ (aug-cc-pVTZ-C); else None."""
    if z > 2 or not _CORE_NAME.search(name):
        return None
    return _CORE_NAME.sub("pV", name)


def source_name(name: str, z: int) -> str:
    """Basis set file that load_basis(name) takes element z from."""
    alt = core_fallback(name, z)
    if alt and z not in _load_file(name):
        try:
            if z in _load_file(alt):
                return alt
        except FileNotFoundError:
            pass
    return name


@lru_cache(maxsize=None)
def load_basis(name: str) -> dict:
    """Read a basis set in BSE JSON format; H and He of p(w)CVXZ sets from pVXZ if the
    file lacks them (see core_fallback).

    General contractions and combined shells (e.g. SP) are split into one shell per
    angular momentum and contraction.

    Returns:
        {Z: [ShellDef]} with segmented shells.
    """
    element_basis = dict(_load_file(name))  # copy: _load_file is cached
    for z in (1, 2):
        alt = source_name(name, z)
        if alt != name:
            element_basis[z] = _load_file(alt)[z]
    return element_basis


@lru_cache(maxsize=None)
def _load_file(name: str) -> dict:
    """{Z: [ShellDef]} of one basis set file."""
    data = _read_json(name)
    element_basis = {}
    for z, el in data["elements"].items():
        shells = []
        for sh in el.get("electron_shells", []):
            ftype = sh["function_type"]
            if ftype not in ("gto", "gto_spherical", "gto_cartesian"):
                raise ValueError(f"{name}: unsupported function type '{ftype}'")
            # plain 'gto' (s, p shells) follows the basis-wide convention
            pure = "gto_cartesian" not in (ftype, *data.get("function_types", []))
            exps = [float(x) for x in sh["exponents"]]
            ams = sh["angular_momentum"]
            for i, coefs in enumerate(sh["coefficients"]):
                l = ams[i] if len(ams) > 1 else ams[0]
                c = [float(x) for x in coefs]
                # drop primitives with zero coefficient (general contractions)
                prim = [(a, b) for a, b in zip(exps, c) if b != 0.0]
                shells.append(ShellDef(l, pure, [a for a, _ in prim], [b for _, b in prim]))
        if shells:
            element_basis[int(z)] = shells
    return element_basis


# --- basis sets per atom ---------------------------------------------------------------

def _key(key):
    """Mapping key as a string; YAML 1.1 reads the element symbol No as false."""
    return "No" if key is False else str(key).strip()


def _parse_name(name):
    """Canonical name "default;El=name;n=name" -> mapping (a plain name: {default: name})."""
    out = {}
    for i, part in enumerate(p.strip() for p in str(name).split(";")):
        key, eq, value = part.partition("=")
        if not eq:
            if i > 0 or not part:
                raise ValueError(f"basis '{name}': expected 'key=name' after the default")
            out["default"] = part
        else:
            out[key.strip()] = value.strip()
    return out


def assign_basis(spec, zs, labels=None, partial=False, what="basis"):
    """Basis name of every atom.

    Args:
        spec: Name, canonical name or mapping {default | element | atom number | label:
            name}; None assigns nothing.
        zs: Nuclear charges of the atoms.
        labels: Atom labels of the input (e.g. 'O1', 'H_a'; None or the plain element
            symbol: no label).
        partial: Atoms not covered get None (else an error).
        what: Input key for error messages.

    Returns:
        List of names (None for uncovered atoms with `partial`).

    Raises:
        ValueError: Unknown key, two keys for an atom, uncovered atom.
    """
    n = len(zs)
    if spec is None:
        mapping = {}
    elif isinstance(spec, dict):
        mapping = {_key(k): v for k, v in spec.items()}
        if len(mapping) < len(spec):
            raise ValueError(f"{what}: duplicate keys")
    else:
        mapping = _parse_name(spec)
    labels = [l if l and l.lower() != element_symbol(z).lower() else None
              for l, z in zip(labels or [None] * n, zs)]
    default, by_z, by_atom, seen = None, {}, {}, set()
    for key, value in mapping.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{what}: expected a basis set name for '{key}', got {value!r}")
        value = value.strip()
        low = key.lower()
        if low in seen:
            raise ValueError(f"{what}: duplicate key '{key}'")
        seen.add(low)
        if low == "default":
            default = value
            continue
        if key.isdigit():
            atoms = [int(key) - 1] if 1 <= int(key) <= n else []
        else:
            atoms = [i for i, l in enumerate(labels) if l == key]
        if atoms:
            for i in atoms:
                if i in by_atom:
                    raise ValueError(f"{what}: atom {i + 1} named twice ('{key}')")
                by_atom[i] = value
            continue
        try:
            z = element_z(key.capitalize())
        except (ValueError, RuntimeError):
            z = None
        if z is None or key.isdigit():
            raise ValueError(f"{what}: '{key}' is no element symbol, atom number (1-{n}) or "
                             f"atom label")
        if z not in zs:
            raise ValueError(f"{what}: no {element_symbol(z)} atom")
        by_z[z] = value
    names = [by_atom.get(i, by_z.get(z, default)) for i, z in enumerate(zs)]
    missing = [f"{i + 1}{element_symbol(z)}" for i, (z, b) in enumerate(zip(zs, names))
               if b is None]
    if missing and not partial:
        raise ValueError(f"{what}: no basis set for atom(s) {', '.join(missing)} (give "
                         f"'default' or the elements)")
    return names


def canonical_name(names, zs):
    """Canonical name of a basis per atom: the most frequent basis (ties: the first in
    atom order), then elements whose atoms deviate ("El=name", the most frequent basis of
    the element, elements in atom order), then single atoms ("n=name")."""
    from collections import Counter
    if len(set(names)) == 1:
        return names[0]
    default = Counter(names).most_common(1)[0][0]
    parts, atoms = [default], []
    for z in dict.fromkeys(zs):
        idx = [i for i, x in enumerate(zs) if x == z]
        count = Counter(names[i] for i in idx)
        el = max(count, key=lambda b: (count[b], b == default))  # ties: the default
        if el != default:
            parts.append(f"{element_symbol(z)}={el}")
        atoms += [i for i in idx if names[i] != el]
    parts += [f"{i + 1}={names[i]}" for i in sorted(atoms)]
    return ";".join(parts)


def basis_names(name, zs):
    """Basis name per atom of a (canonical) basis name."""
    return assign_basis(name, zs)


def atom_shells(name, zs):
    """Shell definitions per atom ([[ShellDef]]) of a (canonical) basis name.

    Raises:
        ValueError: A basis has no functions for an element.
    """
    out = []
    for n, z in zip(basis_names(name, zs), zs):
        lib = load_basis(n)
        if z not in lib:
            raise ValueError(f"basis '{n}' has no functions for {element_symbol(z)}")
        out.append(lib[z])
    return out


def build_basis(mol, spec, labels=None, auxiliary=False) -> BasisSet:
    """Basis set on the atoms of `mol`: a name, a canonical name or a mapping per
    element/atom (see assign_basis); BasisSet.name is the canonical name. auxiliary:
    fitting basis (functions up to the 2-/3-centre limit of libint, l = 8 in the custom build).

    Raises:
        ValueError: Invalid assignment, missing elements, or ECPs (not supported).
    """
    zs = list(mol.atomic_numbers)
    names = assign_basis(spec, zs, labels)
    for b in dict.fromkeys(names):
        ecp = ecp_elements(b) & {z for z, x in zip(zs, names) if x == b}
        if ecp:  # not yet supported: refuse instead of silently running all-electron
            syms = ", ".join(element_symbol(z) for z in sorted(ecp))
            raise ValueError(f"basis '{b}' uses ECPs for {syms}; ECPs are not supported yet")
    name = canonical_name(names, zs)
    if len(set(names)) == 1:
        return BasisSet(mol, load_basis(name), name, auxiliary)
    return BasisSet(mol, atom_shells(name, zs), name, auxiliary)


def contraction_pattern(shells) -> str:
    """Pattern like '(7s4p1d) -> [3s2p1d]' of a list of ShellDef."""
    prim, contr = {}, {}  # distinct exponents / contracted functions per l
    for s in shells:
        contr[s.l] = contr.get(s.l, 0) + 1
        prim.setdefault(s.l, set()).update(s.exponents)
    ls = sorted(contr)
    p = "".join(f"{len(prim[l])}{_LNAMES[l]}" for l in ls)
    c = "".join(f"{contr[l]}{_LNAMES[l]}" for l in ls)
    return f"({p}) -> [{c}]"


def _aux_table():
    """The auxiliary basis table (auxbasis.yaml, first file in the search path)."""
    import yaml
    for d in search_path():
        f = d / "auxbasis.yaml"
        if f.is_file():
            return yaml.safe_load(f.read_text()) or {}, f
    return {}, None


def _auxbasis(kind, basis_name, explicit):
    """Auxiliary basis of a kind ('jk', 'c' or 'cabs') for an orbital basis.

    Order: `explicit` (system key <kind>_auxbasis); an exact entry of the section <kind>
    of auxbasis.yaml (case-insensitive); a file "<basis>-JK" / "<basis>-C" /
    "<basis>-CABS"; for p(w)CVXZ the default of pVXZ; the first matching pattern of the
    section.

    Returns:
        (fitting basis name, how it was chosen).

    Raises:
        ValueError: No fitting basis found.
    """
    from fnmatch import fnmatch
    if explicit:
        return explicit, f"input ({kind}_auxbasis)"
    table, path = _aux_table()
    entries = {str(k): str(v) for k, v in (table.get(kind) or {}).items()}
    for key, value in entries.items():
        if "*" not in key and "?" not in key and key.lower() == basis_name.lower():
            return value, f"default for {basis_name} ({path.name})"
    suffix = "-" + kind.upper()
    try:
        find_basis_file(basis_name + suffix)
        return basis_name + suffix, f"file <basis>{suffix}"
    except FileNotFoundError:
        pass
    alt = core_fallback(basis_name, 1)
    if alt:  # p(w)CVXZ: the default of pVXZ
        try:
            value, how = _auxbasis(kind, alt, None)
            return value, f"{how}, as for {alt}"
        except ValueError:
            pass
    for key, value in entries.items():
        if fnmatch(basis_name.lower(), key.lower()):
            return value, f"pattern '{key}' ({path.name})"
    what = {"jk": "JK fitting", "c": "correlation fitting", "cabs": "CABS"}[kind]
    raise ValueError(f"no {what} basis for '{basis_name}': set {kind}_auxbasis or add an "
                     f"entry to lib/basis/auxbasis.yaml")


def _has_element(name, z):
    """Whether basis `name` (H, He with the p(w)CVXZ fallback) has element z."""
    try:
        return z in load_basis(name)
    except FileNotFoundError:
        return False


def aux_assignment(kind, basis_name, zs, labels=None, explicit=()):
    """Auxiliary basis per atom ('jk', 'c' or 'cabs') for a (canonical) orbital basis.

    Atoms are taken from the `explicit` specifications in order (name or mapping, see
    assign_basis), the rest from the table (_auxbasis) for the orbital basis of each atom;
    H and He of a p(w)CVXZ set take the default of pVXZ if the fitting basis lacks them
    (e.g. cc-pCVTZ-F12-C -> aug-cc-pwCVTZ-C).

    Args:
        explicit: [(spec or None, how)] in order of priority (task option, system key).

    Returns:
        (canonical name, how chosen).

    Raises:
        ValueError: Invalid assignment, no fitting basis for an orbital basis.
    """
    names, hows = [None] * len(zs), [None] * len(zs)
    for spec, how in explicit:
        if spec:
            for i, b in enumerate(assign_basis(spec, zs, labels, True, f"{kind}_auxbasis")):
                if names[i] is None and b is not None:
                    names[i], hows[i] = b, how
    table = {}
    for i, b in enumerate(basis_names(basis_name, zs)):
        if names[i] is None:
            z = zs[i]
            alt = core_fallback(b, z)
            if alt and not _has_element(_auxbasis(kind, b, None)[0], z):
                b = alt  # H, He of p(w)CVXZ: fitting basis of pVXZ if the default lacks them
            if b not in table:
                table[b] = _auxbasis(kind, b, None)
            names[i], hows[i] = table[b]
    hows = list(dict.fromkeys(hows))
    if len(hows) > 1:  # mixed: the sources only (input, task option, table)
        hows = ["per atom: " + ", ".join(dict.fromkeys(
            h if h.startswith(("input", "task")) else "default table" for h in hows))]
    return canonical_name(names, zs), hows[0]


def jk_auxbasis(basis_name, explicit=None):
    """JK fitting basis for an orbital basis (see _auxbasis): (name, how chosen)."""
    return _auxbasis("jk", basis_name, explicit)


def cabs_auxbasis(basis_name, explicit=None):
    """Complementary auxiliary basis (CABS) of F12 methods for an orbital basis (see
    _auxbasis): (name, how chosen); files "<basis>-CABS" come before patterns."""
    return _auxbasis("cabs", basis_name, explicit)


def c_auxbasis(basis_name, explicit=None):
    """Correlation fitting basis for an orbital basis (see _auxbasis): (name, how
    chosen); files "<basis>-C" come before patterns."""
    return _auxbasis("c", basis_name, explicit)