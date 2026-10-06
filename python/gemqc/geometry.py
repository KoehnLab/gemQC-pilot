"""Molecular structure input: XYZ files and text, Z-matrices, stored geometries.

All readers return a Molecule with coordinates in bohr.
"""

import json
from pathlib import Path

import numpy as np

from ._core import Molecule, element_z
from .units import length_factor

_DUMMY = ("x", "q")  # dummy atoms of Z-matrices (construction only)


def _element(label):
    """Element symbol of an atom label like 'O1' or 'H_a'."""
    return "".join(c for c in label.split("_")[0] if c.isalpha())


def _xyz_atoms(lines, fac, mol, where):
    for n, line in lines:
        tok = line.split()
        if len(tok) < 4:
            raise ValueError(f"{where}:{n}: expected 'symbol x y z'")
        try:
            mol.add_atom(element_z(_element(tok[0])), [fac * float(x) for x in tok[1:4]])
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"{where}:{n}: {e}") from None
    return mol


def read_xyz(path, unit="angstrom", name=None) -> Molecule:
    """Read an XYZ file.

    Args:
        path: File name.
        unit: Unit of the coordinates in the file (angstrom, bohr).
        name: Molecule name (default: file stem).

    Returns:
        Molecule with coordinates in bohr.

    Raises:
        ValueError: Malformed file.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    try:
        natoms = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: first line must contain the number of atoms") from None
    if len(lines) < natoms + 2:
        raise ValueError(f"{path}: expected {natoms} atoms, file too short")
    atoms = list(enumerate(lines[2:natoms + 2], start=3))
    return _xyz_atoms(atoms, length_factor(unit), Molecule(name or path.stem), str(path))


def parse_xyz_text(text, unit="angstrom", name="molecule") -> Molecule:
    """Molecule from lines 'symbol x y z' (e.g. inline in the input); blank lines skipped.

    Raises:
        ValueError: Malformed line or no atoms.
    """
    atoms = [(n, l) for n, l in enumerate(text.splitlines(), start=1) if l.strip()]
    if not atoms:
        raise ValueError("geometry: no atoms")
    return _xyz_atoms(atoms, length_factor(unit), Molecule(name), "geometry")


def _place(r, a, d, pi, pj, pk):
    """Atom at distance r from pi, angle a (i, j), dihedral d (i, j, k); radians."""
    bc = pi - pj
    bc /= np.linalg.norm(bc)
    n = np.cross(pj - pk, bc)
    if np.linalg.norm(n) < 1e-8:
        raise ValueError("dihedral reference atoms are collinear")
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    return pi - r * np.cos(a) * bc + r * np.sin(a) * (np.cos(d) * m + np.sin(d) * n)


def parse_zmatrix(text, variables=None, unit="angstrom", name="molecule") -> Molecule:
    """Molecule from a Z-matrix (Molpro/Gaussian style).

    Lines 'atom [i r [j angle [k dihedral]]]'; references i, j, k are 1-based line
    numbers or atom labels of earlier lines; values are numbers or (signed) names of
    `variables`. Distances in `unit`, angles in degrees. The first atom is placed
    at the origin, the second on the z axis, the third in the xz plane. Dummy atoms
    (X, Q) are used for the construction only.

    Raises:
        ValueError: Malformed line, unknown reference or variable.
    """
    variables = {str(k): v for k, v in (variables or {}).items()}
    fac = length_factor(unit)
    labels, pos, mol = [], [], Molecule(name)
    lines = [(n, l.split()) for n, l in enumerate(text.splitlines(), start=1)
             if l.strip()]
    for n, tok in lines:
        where = f"zmatrix line {n}"
        k = len(labels)  # atoms defined so far
        if len(tok) != 1 + 2 * min(k, 3):
            raise ValueError(f"{where}: expected {1 + 2 * min(k, 3)} entries, got {len(tok)}")

        def ref(t):
            if t.isdigit() and 1 <= int(t) <= k:
                return int(t) - 1
            if t in labels:
                return labels.index(t)
            raise ValueError(f"{where}: unknown reference atom '{t}'")

        def value(t):
            sign, key = (-1, t[1:]) if t.startswith("-") else (1, t)
            try:
                return float(t)
            except ValueError:
                if key not in variables:
                    raise ValueError(f"{where}: unknown variable '{key}'") from None
                return sign * float(variables[key])

        refs = [ref(t) for t in tok[1::2]]
        vals = [value(t) for t in tok[2::2]]
        if len(set(refs)) < len(refs):
            raise ValueError(f"{where}: reference atoms must differ")
        if k == 0:
            p = np.zeros(3)
        elif k == 1:
            p = pos[refs[0]] + [0, 0, fac * vals[0]]
        else:
            pi, pj = pos[refs[0]], pos[refs[1]]
            # third atom: dihedral reference off the i-j axis puts it in the xz plane
            pk = pos[refs[2]] if k >= 3 else pj + (
                [1.0, 0, 0] if np.linalg.norm(np.cross(pi - pj, [1.0, 0, 0])) > 1e-8
                else [0, 1.0, 0])
            d = np.radians(vals[2]) if k >= 3 else np.pi
            p = _place(fac * vals[0], np.radians(vals[1]), d, pi, pj, pk)
        labels.append(tok[0])
        pos.append(np.asarray(p, float))
        if _element(tok[0]).lower() not in _DUMMY:
            try:
                mol.add_atom(element_z(_element(tok[0])), list(pos[-1]))
            except (ValueError, RuntimeError) as e:
                raise ValueError(f"{where}: {e}") from None
    if mol.natoms == 0:
        raise ValueError("zmatrix: no atoms")
    return mol


def atom_labels(text, zmatrix=False):
    """Atom labels (first entries) of the lines of an inline geometry or Z-matrix, or of
    an XYZ file (`text` a Path); dummy atoms of Z-matrices skipped."""
    if isinstance(text, Path):
        lines = text.read_text().splitlines()
        lines = lines[2:int(lines[0].split()[0]) + 2]
    else:
        lines = text.splitlines()
    labels = [l.split()[0] for l in lines if l.strip()]
    return [l for l in labels if not (zmatrix and _element(l).lower() in _DUMMY)]


def read_stored(path, system=None, name=None):
    """Geometry of a system in a gemQC HDF5 (restart) or JSON (results) file.

    Coordinates are those of the computational frame of that run (bohr).

    Args:
        path: .h5 or .json file.
        system: System name in the file; None: the only system of the file.
        name: Name of the new molecule (default: the system name).

    Returns:
        (Molecule, system name in the file).

    Raises:
        ValueError: No such system, several systems and none selected.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        stored = {s: (d["atomic_numbers"], d["coordinates_bohr"])
                  for s, d in json.loads(path.read_text()).get("systems", {}).items()}
    else:
        import h5py
        stored = {}
        with h5py.File(path, "r") as f:
            for s, g in f.items():  # /<system>/<task>/molecule; the latest task
                tasks = [t for t in g.values() if "molecule" in t]
                if tasks:
                    t = max(tasks, key=lambda t: t.attrs.get("timestamp", 0))
                    stored[s] = (t["molecule/atomic_numbers"][()].tolist(),
                                 t["molecule/coordinates"][()])
    if system is None:
        if len(stored) != 1:
            raise ValueError(f"{path.name}: select one of the systems "
                             f"{', '.join(stored) or '(none)'} with source_system")
        system = next(iter(stored))
    if system not in stored:
        raise ValueError(f"{path.name}: no system '{system}' (stored: "
                         f"{', '.join(stored) or 'none'})")
    zs, xyz = stored[system]
    mol = Molecule(name or system)
    for z, r in zip(zs, np.asarray(xyz, float)):
        mol.add_atom(int(z), list(r))
    return mol, system


def distance_range(mol: Molecule):
    """Shortest and longest interatomic distance (bohr) with their atom pairs.

    Returns:
        ((r_min, i, j), (r_max, i, j)) with 0-based atoms, or None for one atom.
    """
    X = mol.coordinates
    if len(X) < 2:
        return None
    i, j = np.triu_indices(len(X), k=1)
    r = np.linalg.norm(X[i] - X[j], axis=1)
    a, b = r.argmin(), r.argmax()
    return (r[a], i[a], j[a]), (r[b], i[b], j[b])


def coordinates_in(mol: Molecule, unit="angstrom") -> np.ndarray:
    """Coordinates of `mol` (natoms, 3) converted from bohr to `unit`."""
    return mol.coordinates / length_factor(unit)
