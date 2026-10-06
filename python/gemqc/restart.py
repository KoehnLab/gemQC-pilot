"""Restart data on HDF5: molecule, basis definition, SCF energy and orbitals.

Layout of the file (one per run), group /<system>/<task>/::

    molecule/  atomic_numbers, coordinates (bohr); attrs charge, multiplicity
    basis/     attrs name; per element 'Z<z>': l, pure, nprim, exponents,
               coefficients (as in the basis library, unnormalized)
    attrs energy, converged, iterations, nbf, timestamp, symmetry, method;
    datasets C, eps, occ, D, orbital_labels (n.k); UHF: C, eps, occ,
    orbital_labels are alpha, plus C_b, eps_b, occ_b, orbital_labels_b,
    D_spin, C_nat, occ_nat, natural_labels, attrs s2, s2_exact; ROHF: D_spin;
    attrs properties_origin, dipole, quadrupole (au, centre of mass, traceless);
    (coordinates and orbitals refer to the standard orientation)
"""

import time

import h5py
import numpy as np

from . import __version__, _core
from .basislib import basis_names, load_basis


def _write_shells(g, shells):
    g["l"] = [s.l for s in shells]
    g["pure"] = [s.pure for s in shells]
    g["nprim"] = [len(s.exponents) for s in shells]
    g["exponents"] = np.concatenate([s.exponents for s in shells])
    g["coefficients"] = np.concatenate([s.coefficients for s in shells])


def _write_basis(grp, basis_name, zs):
    """Basis definition: groups Z<z> per element; with several basis sets (canonical
    name, see basislib) groups Z<z>_<k> per element and basis k, and the basis index
    of each atom (dataset atom_basis)."""
    grp.attrs["name"] = basis_name
    names = basis_names(basis_name, list(zs))
    distinct = list(dict.fromkeys(names))
    if len(distinct) == 1:
        lib = load_basis(basis_name)
        for z in sorted(set(zs)):
            _write_shells(grp.create_group(f"Z{z}"), lib[z])
        return
    grp["atom_basis"] = [distinct.index(n) for n in names]
    for k, b in enumerate(distinct):
        lib = load_basis(b)
        for z in sorted({z for z, n in zip(zs, names) if n == b}):
            _write_shells(grp.create_group(f"Z{z}_{k}"), lib[z])


def _read_shells(g):
    ls, pure, nprim = g["l"][()], g["pure"][()], g["nprim"][()]
    exps, coefs = g["exponents"][()], g["coefficients"][()]
    shells, k = [], 0
    for l, p, n in zip(ls, pure, nprim):
        shells.append(_core.ShellDef(int(l), bool(p), exps[k:k + n].tolist(),
                                     coefs[k:k + n].tolist()))
        k += n
    return shells


def _read_basis(grp, zs):
    """Basis stored in the file: {Z: [ShellDef]}, or [[ShellDef]] per atom (atoms `zs`)
    for several basis sets."""
    if "atom_basis" in grp:
        return [_read_shells(grp[f"Z{z}_{k}"]) for z, k in zip(zs, grp["atom_basis"][()])]
    return {int(key[1:]): _read_shells(g) for key, g in grp.items()}


def _new_group(f, name, mol, basis_name):
    """(Re)create group `name` with the molecule and basis definition."""
    f.attrs["program"] = f"gemQC {__version__}"
    if name in f:
        del f[name]
    g = f.create_group(name)
    m = g.create_group("molecule")
    m["atomic_numbers"] = mol.atomic_numbers
    m["coordinates"] = mol.coordinates
    m.attrs["charge"] = mol.charge
    m.attrs["multiplicity"] = mol.multiplicity
    _write_basis(g.create_group("basis"), basis_name, mol.atomic_numbers)
    return g


def save_orbitals(path, system, task, mol, basis_name, data, attrs):
    """Store an orbital set under /<system>/<task>.

    Args:
        path: HDF5 file (created or updated; the group is replaced).
        system: System name.
        task: Group name.
        mol: Molecule (stored with the data).
        basis_name: Basis set name (its definition is stored).
        data: Datasets {name: array or list of str}.
        attrs: Attributes {name: value}.
    """
    with h5py.File(path, "a") as f:
        g = _new_group(f, f"{system}/{task}", mol, basis_name)
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                value = np.array(value, dtype="S")
            g[key] = value
        for key, value in attrs.items():
            g.attrs[key] = value
        g.attrs["timestamp"] = time.time()


def save_wavefunction(path, system, task, wf, symmetry="C1", labels=None, attrs=None, data=None):
    """Store an SCF wavefunction under /<system>/<task> (replaces an old one).

    Args:
        path: HDF5 file.
        system: System name.
        task: Group name ('scf', 'uhf').
        wf: Wavefunction.
        symmetry: Computational point group.
        labels: Orbital labels n.k (UHF: dict with keys alpha, beta, natural).
        attrs: Further attributes (e.g. dipole, quadrupole).
        data: Further datasets.
    """
    with h5py.File(path, "a") as f:
        g = _new_group(f, f"{system}/{task}", wf.molecule, wf.basis.name)
        keys = ["C", "eps", "occ", "D"]
        if wf.unrestricted:
            keys += ["C_b", "eps_b", "occ_b", "D_spin", "C_nat", "occ_nat"]
            g.attrs["s2"] = wf.s2
            g.attrs["s2_exact"] = wf.s2_exact
        elif wf.method == "ROHF":
            keys.append("D_spin")
        for key in keys:
            g[key] = getattr(wf, key)
        g.attrs["method"] = wf.method
        g.attrs["energy"] = wf.energy
        g.attrs["converged"] = wf.converged
        g.attrs["iterations"] = wf.iterations
        g.attrs["nbf"] = wf.basis.nbf
        g.attrs["timestamp"] = time.time()
        g.attrs["symmetry"] = symmetry
        for key, value in (attrs or {}).items():
            g.attrs[key] = value
        for key, value in (data or {}).items():
            g[key] = value
        if isinstance(labels, dict):
            for key, name in (("alpha", "orbital_labels"), ("beta", "orbital_labels_b"),
                              ("natural", "natural_labels")):
                g[name] = np.array(labels[key], dtype="S")
        elif labels is not None:
            g["orbital_labels"] = np.array(labels, dtype="S")


def _matches(g, mol, geom_tol):
    """Stored molecule has the same atoms, charge, multiplicity (and geometry)."""
    m = g["molecule"]
    if list(m["atomic_numbers"][()]) != list(mol.atomic_numbers):
        return False
    if (m.attrs["charge"], m.attrs["multiplicity"]) != (mol.charge, mol.multiplicity):
        return False
    return geom_tol is None or np.abs(m["coordinates"][()] - mol.coordinates).max() < geom_tol


def find_restart(path, system, tasks, mol, geom_tol=1e-6):
    """Group name of the best stored wavefunction for `mol`.

    Candidates are converged results of the given tasks with the same atoms, charge,
    multiplicity and geometry. Preference: the same system (e.g. a previous run),
    the first task, then the most recently written one.

    Args:
        path: HDF5 file.
        system: System name.
        tasks: Task groups to consider, e.g. ("uhf", "scf").
        mol: Molecule.
        geom_tol: Tolerance of the coordinates (bohr); None: geometry not compared.

    Returns:
        '<system>/<task>' or None.
    """
    try:
        f = h5py.File(path, "r")
    except (FileNotFoundError, OSError):
        return None
    with f:
        found = []
        for name, grp in f.items():
            for task in tasks:
                if task not in grp:
                    continue
                g = grp[task]
                if g.attrs.get("converged", False) and _matches(g, mol, geom_tol):
                    found.append((name == system, task == tasks[0], g.attrs.get("timestamp", 0.0),
                                  f"{name}/{task}"))
    return max(found)[3] if found else None


def restart_guess(path, group, mol, basis, out=None, spin=False, shells=False):
    """Guess densities from stored orbitals, projected onto another basis or geometry.

    Args:
        path: HDF5 file.
        group: '<system>/<task>'.
        mol: Molecule (same atoms, charge and multiplicity as stored).
        basis: Target BasisSet.
        out: Output printer, or None.
        spin: Return [alpha, beta] densities from UHF data.
        shells: Return [closed, open] densities (unit occupation) from ROHF data.

    Returns:
        List of densities: [total], or per spin or shell as requested and available.

    Raises:
        FileNotFoundError: No such group.
        ValueError: Different atoms, charge or multiplicity.
    """
    with h5py.File(path, "r") as f:
        if group not in f:
            raise FileNotFoundError(f"{path}: no restart data for '{group}'")
        g = f[group]
        if not _matches(g, mol, None):
            raise ValueError(f"{path} [{group}]: different atoms, charge or multiplicity")
        old_name = g["basis"].attrs["name"]
        old_basis = _core.BasisSet(mol, _read_basis(g["basis"], list(mol.atomic_numbers)),
                                   old_name)
        sets = [(g["C"][()], g["occ"][()])]
        if "C_b" in g:
            sets.append((g["C_b"][()], g["occ_b"][()]))
        rohf = g.attrs.get("method", "") == "ROHF"
    if shells and rohf:  # closed and open shells as separate sets
        C, occ = sets[0]
        sets = [(C, np.where(occ > 1.5, 1.0, 0.0)), (C, np.where((occ > 0.5) & (occ < 1.5), 1.0, 0.0))]
    # project occupied orbitals, then re-orthonormalize them (Loewdin)
    S = _core.overlap(basis)
    S12 = _core.overlap(basis, old_basis)
    dens = []
    for C, occ in sets:
        occ_idx = occ > 0
        Co = np.linalg.solve(S, S12 @ C[:, occ_idx])
        w, u = np.linalg.eigh(Co.T @ S @ Co)
        Co = Co @ u @ np.diag(w ** -0.5) @ u.T
        dens.append((Co * occ[occ_idx]) @ Co.T)
    if out:
        out(f"  restart guess from {path.name} [{group}], basis {old_name} -> {basis.name}")
    return dens if (spin or (shells and rohf)) else [sum(dens)]
