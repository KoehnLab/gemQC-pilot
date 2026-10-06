"""Molecular point-group symmetry (D2h and subgroups for computations).

The full point group is detected for information; computations use its
largest subgroup of D2h ("computational group"). The molecule is reoriented
internally into the standard orientation of that group (Molpro conventions),
where every operation is a sign change of Cartesian axes. How far the input
geometry may be changed is set by `modify`::

    off     input frame and coordinates kept (symmetry only if the input axes
            about the input origin carry it)
    align   rotation and translation only; symmetry within a tight tolerance
    adapt   additionally symmetrized within a user tolerance

Irreps are numbered as in Molpro.
"""

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np


# D2h operations as sign changes of (x, y, z) in the standard orientation
OPS = {"E": (1, 1, 1), "C2z": (-1, -1, 1), "C2y": (-1, 1, -1), "C2x": (1, -1, -1),
       "i": (-1, -1, -1), "sxy": (1, 1, -1), "sxz": (1, -1, 1), "syz": (-1, 1, 1)}
OP_LABELS = {"E": "E", "C2z": "C2(z)", "C2y": "C2(y)", "C2x": "C2(x)", "i": "i",
             "sxy": "sig(xy)", "sxz": "sig(xz)", "syz": "sig(yz)"}

# computational groups: operations and irreps in Molpro order; an irrep is given
# by the (x, y, z) parity of a function transforming like it (e.g. B1 ~ x)
GROUPS = {
    "D2h": (list(OPS), [("Ag", (0, 0, 0)), ("B3u", (1, 0, 0)), ("B2u", (0, 1, 0)),
                        ("B1g", (1, 1, 0)), ("B1u", (0, 0, 1)), ("B2g", (1, 0, 1)),
                        ("B3g", (0, 1, 1)), ("Au", (1, 1, 1))]),
    "C2v": (["E", "C2z", "sxz", "syz"], [("A1", (0, 0, 0)), ("B1", (1, 0, 0)),
                                        ("B2", (0, 1, 0)), ("A2", (1, 1, 0))]),
    "D2": (["E", "C2z", "C2y", "C2x"], [("A", (0, 0, 0)), ("B3", (1, 0, 0)),
                                       ("B2", (0, 1, 0)), ("B1", (0, 0, 1))]),
    "C2h": (["E", "C2z", "i", "sxy"], [("Ag", (0, 0, 0)), ("Au", (0, 0, 1)),
                                       ("Bu", (1, 0, 0)), ("Bg", (1, 0, 1))]),
    "C2": (["E", "C2z"], [("A", (0, 0, 0)), ("B", (1, 0, 0))]),
    "Cs": (["E", "sxy"], [("A'", (0, 0, 0)), ("A''", (0, 0, 1))]),
    "Ci": (["E", "i"], [("Ag", (0, 0, 0)), ("Au", (1, 1, 1))]),
    "C1": (["E"], [("A", (0, 0, 0))]),
}
# preference for the automatic choice (larger order first; C2v before D2)
_PREFERENCE = ["D2h", "C2v", "D2", "C2h", "C2", "Cs", "Ci", "C1"]

# geometry modifications and their default tolerance (bohr); off and align do not
# symmetrize, so only deviations at the level of rounding errors are accepted
MODIFY = ("off", "align", "adapt")
TOLERANCE = {"off": 1e-10, "align": 1e-10, "adapt": 1e-3}

# functions listed in the irrep table
_FUNCTIONS = [("s", (0, 0, 0)), ("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)),
              ("xy", (1, 1, 0)), ("xz", (1, 0, 1)), ("yz", (0, 1, 1)),
              ("x2", (0, 0, 0)), ("y2", (0, 0, 0)), ("z2", (0, 0, 0)), ("xyz", (1, 1, 1))]


# angular tolerance (|cos|) for parallel / perpendicular symmetry elements; element
# directions from slightly distorted geometries are only accurate to ~ tol / r,
# while distinct elements (n <= 12) differ by >= 15 degrees
_ANG = 1e-2


def character(op, parity):
    """Character (+1/-1) of D2h operation `op` for a function of (x, y, z) parity."""
    return int(np.prod([s if p else 1 for s, p in zip(OPS[op], parity)]))


@dataclass
class Symmetry:
    """Symmetry of a molecule and its computational frame.

    Attributes:
        point_group: Full point group (Schoenflies), for information.
        group: Computational group (largest D2h subgroup).
        tolerance: Tolerance of the detection (bohr).
        ops: Operation names of the computational group.
        atom_map: (nops, natoms) image atom of each atom under each operation.
        rotation: Rows are the computational axes in the input frame.
        origin: Input-frame point that becomes the origin.
        max_displacement: Largest deviation of an atom from exact symmetry (bohr);
            removed by the symmetrization for modify='adapt'.
        orientation_note: How the frame was chosen (input kept or standard).
        irreps: [(name, parity)] in Molpro order.
        modify: Allowed geometry modification ('off', 'align', 'adapt').
        nuclear_point_group: Point group of the nuclei if the basis sets per atom lower
            it (point_group is then that of nuclei and basis sets), else "".
    """
    point_group: str          # full point group (Schoenflies)
    group: str                # computational group (D2h subgroup)
    tolerance: float          # bohr
    ops: list                 # operation names of the computational group
    atom_map: np.ndarray      # (nops, natoms): image of each atom
    rotation: np.ndarray      # rows: standard axes in the input frame
    origin: np.ndarray        # input-frame point that becomes the origin
    max_displacement: float = 0.0  # deviation from exact symmetry (bohr)
    orientation_note: str = ""
    irreps: list = field(default_factory=list)   # [(name, parity)]
    modify: str = "adapt"
    nuclear_point_group: str = ""  # if lowered by the basis sets per atom

    @property
    def order(self):
        """Order of the computational group."""
        return len(self.ops)

    def to_standard(self, xyz_input):
        """Input-frame coordinates -> standard orientation."""
        return (np.asarray(xyz_input) - self.origin) @ self.rotation.T

    def to_input(self, xyz_std):
        """Standard orientation -> input frame."""
        return np.asarray(xyz_std) @ self.rotation + self.origin


# --- geometry analysis ---------------------------------------------------------

def _rotation(axis, angle):
    """Rotation matrix about unit vector `axis` (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _mirror(normal):
    return np.eye(3) - 2 * np.outer(normal, normal)


class _Analyzer:
    """Tests symmetry operations on centred coordinates within a tolerance."""

    def __init__(self, Z, X, tol):
        self.Z, self.X, self.tol = np.asarray(Z), np.asarray(X), tol

    def atom_map(self, R):
        """Image atom of each atom under R, or None if R is no symmetry operation."""
        Y = self.X @ np.asarray(R).T
        d = np.linalg.norm(Y[:, None, :] - self.X[None, :, :], axis=2)
        d[self.Z[:, None] != self.Z[None, :]] = np.inf
        j = d.argmin(axis=1)
        if d[np.arange(len(j)), j].max() > self.tol or len(set(j)) < len(j):
            return None
        return j

    def is_op(self, R):
        return self.atom_map(R) is not None

    def classes(self):
        """Groups of atoms with equal Z and distance from the centre."""
        r = np.linalg.norm(self.X, axis=1)
        order = np.lexsort((r, self.Z))
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if self.Z[a] == self.Z[b] and abs(r[b] - r[a]) < self.tol:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        return [g for g in groups if r[g[0]] > self.tol]  # off-centre atoms only

    def directions(self, extra=()):
        """Candidate axis / mirror-normal directions (unique, unit length)."""
        cands = list(extra)
        for cls in self.classes():
            P = self.X[cls]
            for a in range(len(cls)):
                cands.append(P[a])
                for b in range(a + 1, len(cls)):
                    cands += [P[a] + P[b], P[a] - P[b], np.cross(P[a], P[b])]
        uniq = []
        for v in cands:
            n = np.linalg.norm(v)
            if n < self.tol:
                continue
            v = v / n
            if not any(abs(abs(v @ u) - 1) < 1e-4 for u in uniq):
                uniq.append(v)
        return uniq


def _inertia_axes(Z, X):
    w = np.asarray(Z, float)
    I = np.eye(3) * np.sum(w * np.sum(X ** 2, axis=1)) - (X * w[:, None]).T @ X
    return np.linalg.eigh(I)[1].T  # rows


def _point_group(an, dirs):
    """Schoenflies symbol of the full point group; also C2 axes and mirror normals."""
    X = an.X
    has_i = an.is_op(-np.eye(3))
    if len(X) == 1:
        return "Kh", dirs, dirs
    if np.linalg.svd(X, compute_uv=False)[1] < an.tol:  # linear
        return ("Dinfh" if has_i else "Cinfv"), [], []
    nmax = min(12, max([len(c) for c in an.classes()] + [2]))
    axes = []  # (direction, highest order)
    for d in dirs:
        n = max([k for k in range(2, nmax + 1) if an.is_op(_rotation(d, 2 * np.pi / k))],
                default=1)
        if n >= 2:
            axes.append((d, n))
    mirrors = [d for d in dirs if an.is_op(_mirror(d))]
    c2 = [d for d, n in axes if n % 2 == 0]

    def improper(d, n):  # S_n about d
        return an.is_op(_mirror(d) @ _rotation(d, 2 * np.pi / n))

    high = [a for a in axes if a[1] >= 3]
    if len(high) >= 2:
        ns = {n for _, n in high}
        if 5 in ns:
            return ("Ih" if has_i else "I"), c2, mirrors
        if 4 in ns:
            return ("Oh" if has_i else "O"), c2, mirrors
        return ("Th" if has_i else "Td" if mirrors else "T"), c2, mirrors
    if not axes:
        return ("Cs" if mirrors else "Ci" if has_i else "C1"), c2, mirrors
    nprin = max(n for _, n in axes)
    cands = [d for d, n in axes if n == nprin]
    d = next((c for c in cands if improper(c, 2 * nprin)), cands[0])  # e.g. S4 axis of D2d
    perp_c2 = [a for a in c2 if abs(a @ d) < _ANG]
    sigma_h = any(abs(abs(m @ d) - 1) < _ANG for m in mirrors)
    sigma_v = any(abs(m @ d) < _ANG for m in mirrors)
    if perp_c2:
        g = f"D{nprin}h" if sigma_h else f"D{nprin}d" if sigma_v else f"D{nprin}"
    elif sigma_h:
        g = f"C{nprin}h"
    elif sigma_v:
        g = f"C{nprin}v"
    elif improper(d, 2 * nprin):
        g = f"S{2 * nprin}"
    else:
        g = f"C{nprin}"
    return g, c2, mirrors


def _frames(c2, mirrors, inertia):
    """Candidate orthonormal frames (rows x, y, z) spanned by symmetry elements."""
    special = c2 + mirrors
    frames = [inertia]
    for a in special:
        for b in special:
            if abs(a @ b) < _ANG:
                b = b - (b @ a) * a
                b = b / np.linalg.norm(b)
                frames.append(np.array([a, b, np.cross(a, b)]))
        # one special direction: complete with the inertia axis most perpendicular
        e = max(inertia, key=lambda v: np.linalg.norm(v - (v @ a) * a))
        e = e - (e @ a) * a
        e /= np.linalg.norm(e)
        frames.append(np.array([e, np.cross(a, e), a]))
    return frames


def _convention_score(group, F, X):
    """Tie-breaking score of an oriented frame (higher is better): Mulliken/Molpro
    conventions - planar molecules in the yz plane, z through most atoms."""
    Y = X @ F.T
    tol = 1e-3
    in_yz = int(np.sum(np.abs(Y[:, 0]) < tol))
    on_z = int(np.sum(np.linalg.norm(Y[:, :2], axis=1) < tol))
    if group == "C2v":
        return (in_yz,)
    if group in ("D2h", "D2"):
        # long axis (smallest spread perpendicular) along z as final tie-breaker
        return (in_yz, on_z, -float(np.sum(Y[:, 0] ** 2 + Y[:, 1] ** 2)))
    return (0,)


def analyze(Z, xyz, tolerance=None, request="auto", modify="adapt", types=None):
    """Point group, computational group and computational frame of a molecule.

    The input frame is kept if its axes carry the computational group (about the
    input origin, else about the centre of nuclear charge), as in Molpro; else a
    standard orientation is chosen (not for modify='off').

    Args:
        Z: Nuclear charges.
        xyz: Coordinates (natoms, 3), bohr, input frame.
        tolerance: Tolerance of the detection (bohr); None: TOLERANCE[modify].
        request: 'auto', 'nosym'/'c1', or a D2h subgroup name, e.g. 'c2v'.
        modify: 'off' (input frame kept), 'align' (rotation and translation) or
            'adapt' (also symmetrization).
        types: Atom types for the equivalence of atoms (e.g. element and basis set);
            None: Z. The centre of charge and the inertia axes use Z.

    Returns:
        (Symmetry, coordinates in the computational frame or None if unchanged).

    Raises:
        ValueError: Unknown request or modify, requested group not present.
    """
    if types is not None and len(set(zip(Z, types))) > len(set(Z)):
        try:
            sym, xyz_new = _analyze(Z, xyz, tolerance, request, modify, types)
        except ValueError as e:
            try:  # the nuclei alone carry the requested group: the basis sets lower it
                _analyze(Z, xyz, tolerance, request, modify, Z)
            except ValueError:
                raise e from None
            raise ValueError(f"the basis sets per atom do not allow {request} symmetry "
                             f"(symmetry-equivalent atoms with different basis sets)") from None
        nuclear = _analyze(Z, xyz, tolerance, "nosym", modify, Z)[0].point_group
        if nuclear != sym.point_group:
            sym.nuclear_point_group = nuclear
        return sym, xyz_new
    return _analyze(Z, xyz, tolerance, request, modify, Z)


def _analyze(Z, xyz, tolerance, request, modify, T):
    """analyze with atom types T for the equivalence of atoms."""
    if modify not in MODIFY:
        raise ValueError(f"unknown modify '{modify}' ({', '.join(MODIFY)})")
    tolerance = TOLERANCE[modify] if tolerance is None else tolerance
    Z, T = np.asarray(Z), np.asarray(T)
    xyz = np.asarray(xyz, float)
    origin = (Z[:, None] * xyz).sum(axis=0) / Z.sum()  # centre of nuclear charge
    X = xyz - origin
    an = _Analyzer(T, X, tolerance)
    inertia = _inertia_axes(Z, X)
    dirs = an.directions(extra=list(inertia))
    full, c2, mirrors = _point_group(an, dirs)
    if full in ("Dinfh", "Cinfv", "Kh"):  # every perpendicular direction qualifies
        c2 = mirrors = []

    req = request.strip().lower()
    if req in ("nosym", "c1"):
        sym, _ = _no_symmetry(full, tolerance, len(Z), modify)
        sym.orientation_note = "input orientation kept (symmetry switched off)"
        return sym, None
    names = {g.lower(): g for g in GROUPS}
    if req != "auto" and req not in names:
        raise ValueError(f"unknown symmetry '{request}' (auto, nosym, "
                         + ", ".join(g.lower() for g in _PREFERENCE) + ")")
    wanted = _PREFERENCE if req == "auto" else [names[req]]
    if modify == "off":
        return _input_frame(T, xyz, full, tolerance, wanted, request)

    best = None
    for F in _frames(c2, mirrors, inertia):
        for perm in permutations(range(3)):
            G = F[list(perm)]
            if np.linalg.det(G) < 0:
                G = G * np.array([[1], [1], [-1]])
            present = {op for op, s in OPS.items() if an.is_op(G.T @ np.diag(s) @ G)}
            for rank, g in enumerate(wanted):
                if set(GROUPS[g][0]) <= present:
                    key = (-rank, _convention_score(g, G, X))
                    if best is None or key > best[0]:
                        best = (key, g, G)
                    break
    if best is None:
        raise ValueError(f"molecule does not have {request} symmetry "
                         f"(point group {full}, tolerance {tolerance:g} bohr)")
    _, group, G = best
    if group == "C1":
        return _no_symmetry(full, tolerance, len(Z), modify)
    ops = GROUPS[group][0]
    # as Molpro: keep the input frame if it already carries the group (axes and
    # origin), else the input axes about the centre of charge; otherwise the
    # standard orientation found above
    note = "standard: " + _orientation_note(group)
    for org, text in ((np.zeros(3), "input orientation kept"),
                      (origin, "input axes kept, origin moved to the centre of charge")):
        an_in = _Analyzer(T, xyz - org, tolerance)
        if all(an_in.is_op(np.diag(OPS[op])) for op in ops):
            G, origin, X, note = np.eye(3), org, xyz - org, text
            break
    return _finish(T, X @ G.T, full, group, tolerance, G, origin, note, modify)


def _input_frame(Z, xyz, full, tolerance, wanted, request):
    """modify='off': the first wanted group whose operations the input axes carry."""
    an = _Analyzer(Z, xyz, tolerance)
    present = {op for op, s in OPS.items() if an.is_op(np.diag(s))}
    group = next((g for g in wanted if set(GROUPS[g][0]) <= present), None)
    if group is None:
        raise ValueError(f"molecule does not have {request} symmetry in its input frame "
                         f"(point group {full}; modify: off keeps the input frame)")
    if group == "C1":
        return _no_symmetry(full, tolerance, len(Z), "off")
    return _finish(Z, xyz, full, group, tolerance, np.eye(3), np.zeros(3),
                   "input orientation kept (modify: off)", "off")


def _finish(Z, Xs, full, group, tolerance, G, origin, note, modify):
    """Symmetry object and coordinates in the computational frame (symmetrized
    for modify='adapt'; else unchanged, None for 'off')."""
    ops = GROUPS[group][0]
    an_s = _Analyzer(Z, Xs, tolerance)
    amap = np.array([an_s.atom_map(np.diag(OPS[op])) for op in ops])
    # symmetrize: average each atom over the images of its partners
    sym = np.zeros_like(Xs)
    for k, op in enumerate(ops):
        sym[:] += Xs[amap[k]] * np.array(OPS[op])
    sym /= len(ops)
    sym_info = Symmetry(full, group, tolerance, ops, amap, G, origin,
                        float(np.abs(sym - Xs).max()), note, GROUPS[group][1], modify)
    return sym_info, {"off": None, "align": Xs, "adapt": sym}[modify]


def _no_symmetry(full, tolerance, natoms, modify="adapt"):
    """C1: input orientation is kept."""
    sym = Symmetry(full, "C1", tolerance, ["E"], np.arange(natoms)[None, :], np.eye(3),
                   np.zeros(3), 0.0, "input orientation kept", GROUPS["C1"][1], modify)
    return sym, None


def _orientation_note(group):
    return {"D2h": "C2 axes along x, y, z (planar molecules in yz plane)",
            "D2": "C2 axes along x, y, z",
            "C2v": "C2 axis along z, mirror planes xz and yz (planar molecules in yz)",
            "C2h": "C2 axis along z, mirror plane xy",
            "C2": "C2 axis along z", "Cs": "mirror plane xy",
            "Ci": "inversion centre at origin"}.get(group, "")


def symmetrize_molecule(mol, tolerance=None, request="auto", modify="adapt", types=None):
    """Detect the symmetry and move `mol` into the computational frame.

    Args:
        mol: Molecule; its coordinates are replaced (unless modify='off').
        tolerance: Tolerance of the detection (bohr); None: TOLERANCE[modify].
        request: 'auto', 'nosym' or a D2h subgroup name.
        modify: 'off', 'align' or 'adapt' (symmetrized), see analyze.
        types: Atom types for the equivalence of atoms (e.g. element and basis set).

    Returns:
        Symmetry.
    """
    sym, xyz = analyze(mol.atomic_numbers, mol.coordinates, tolerance, request, modify, types)
    if xyz is not None:
        mol.coordinates = xyz
    return sym


def unique_atoms(sym):
    """Indices of symmetry-unique atoms (the first of each set of equivalent atoms)."""
    seen, uniq = set(), []
    for a in range(sym.atom_map.shape[1]):
        if a not in seen:
            uniq.append(a)
            seen.update(int(b) for b in sym.atom_map[:, a])
    return uniq


def print_symmetry(sym, natoms, out):
    """Print the point group, orientation and irrep numbering (Molpro conventions)."""
    out.section("Symmetry")
    full = {"Dinfh": "D(inf)h", "Cinfv": "C(inf)v", "Kh": "K(h) (atom)"}.get(
        sym.point_group, sym.point_group)
    out.value("point group", f"{full}   (tolerance {sym.tolerance:.1e} bohr)")
    if sym.nuclear_point_group:
        out.value("lowered from", f"{sym.nuclear_point_group} (nuclei) by the basis sets per "
                  f"atom")
    how = "switched off" if "switched off" in sym.orientation_note else "largest subgroup of D2h"
    out.value("computational group", f"{sym.group}   ({how})")
    if sym.group == "C1":
        out.value("orientation", sym.orientation_note)
        return
    out.value("orientation", sym.orientation_note)
    if sym.modify == "adapt":
        out.value("symmetrization", f"max. displacement {sym.max_displacement:.1e} bohr")
    else:
        out.value("deviation from symmetry", f"max. {sym.max_displacement:.1e} bohr "
                  f"(not symmetrized, modify: {sym.modify})")
    out.value("symmetry-unique atoms", f"{len(unique_atoms(sym))} of {natoms}")
    out()
    out(f"  Irreducible representations of {sym.group} (Molpro numbering; "
        f"orbital label n.k = n-th orbital of irrep k):")
    ops = sym.ops
    head = "    no.  irrep  " + "".join(f"{OP_LABELS[o]:>8s}" for o in ops) + "    functions"
    out(head)
    for k, (name, par) in enumerate(sym.irreps, start=1):
        funcs = [f for f, p in _FUNCTIONS
                 if all(character(o, p) == character(o, par) for o in ops)]
        chars = "".join(f"{character(o, par):>8d}" for o in ops)
        out(f"    {k:3d}  {name:<5s}  {chars}    {', '.join(funcs)}")


# --- basis functions and orbitals ------------------------------------------------

def _function_parities(l, pure):
    """(x, y, z) parities of the functions of a shell in libint order."""
    if pure:  # real solid harmonics, m = -l..l (m<0: sin-type, m>0: cos-type)
        res = []
        for m in range(-l, l + 1):
            am = abs(m)
            px = (am % 2) if m >= 0 else ((am + 1) % 2)
            res.append((px, int(m < 0), (l + am) % 2))
        return res
    res = []
    for i in range(l, -1, -1):  # x^i y^j z^k, libint (CCA) order
        for j in range(l - i, -1, -1):
            res.append((i % 2, j % 2, (l - i - j) % 2))
    return res


def ao_operations(basis, sym):
    """AO representation of the operations of the computational group.

    Returns:
        [(perm, sign)] per operation with R phi_mu = sign[mu] * phi_perm[mu].
    """
    shells_of = {}
    for s in range(basis.nshells):
        shells_of.setdefault(basis.shell_to_atom[s], []).append(s)
    offsets = basis.shell_offsets
    reps = []
    for k, op in enumerate(sym.ops):
        perm = np.empty(basis.nbf, dtype=int)
        sign = np.empty(basis.nbf)
        for atom, shells in shells_of.items():
            image = shells_of[int(sym.atom_map[k, atom])]
            for s, t in zip(shells, image):
                sh = basis.shell(s)
                for f, par in enumerate(_function_parities(sh["l"], sh["pure"])):
                    perm[offsets[s] + f] = offsets[t] + f
                    sign[offsets[s] + f] = character(op, par)
        reps.append((perm, sign))
    return reps


def subgroup(sym, irrep):
    """Subgroup that keeps functions of an irrep.

    Args:
        sym: Symmetry.
        irrep: Irrep index (0-based).

    Returns:
        (operations with character +1, name of the subgroup).
    """
    ops = [op for op in sym.ops if character(op, sym.irreps[irrep][1]) == 1]
    n_c2 = sum(op.startswith("C2") for op in ops)
    n_sig = sum(op.startswith("s") for op in ops)
    name = {1: "C1", 8: "D2h"}.get(len(ops))
    if len(ops) == 2:
        name = "C2" if n_c2 else ("Cs" if n_sig else "Ci")
    elif len(ops) == 4:
        name = "D2" if n_c2 == 3 else ("C2v" if n_sig == 2 else "C2h")
    return ops, name


def character_table(sym):
    """characters[irrep][op] of the computational group (Molpro order)."""
    return [[float(character(op, par)) for op in sym.ops] for _, par in sym.irreps]


def _apply(rep, C):
    perm, sign = rep
    out = np.empty_like(C)
    out[perm] = sign[:, None] * C
    return out


def so_basis(basis, sym):
    """Symmetry-adapted functions.

    Returns:
        Per irrep a matrix (nbf, n_k) whose columns are normalized +- combinations of
        symmetry-equivalent AOs (projected AOs).
    """
    reps = ao_operations(basis, sym)
    chars = [[character(op, par) for op in sym.ops] for _, par in sym.irreps]
    blocks = [[] for _ in sym.irreps]
    done = np.zeros(basis.nbf, dtype=bool)
    for mu in range(basis.nbf):
        if done[mu]:
            continue
        orbit = {int(perm[mu]) for perm, _ in reps}
        for k, chi in enumerate(chars):
            v = np.zeros(basis.nbf)
            for c, (perm, sign) in zip(chi, reps):
                v[perm[mu]] += c * sign[mu]
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                blocks[k].append(v / norm)
        done[list(orbit)] = True
    return [np.array(b).T if b else np.zeros((basis.nbf, 0)) for b in blocks]


def scf_symmetry(basis, sym, occupation=None, freeze_occupation=0, skeleton=True,
                 occupation_beta=None, occupation_open=None):
    """ScfSymmetry for the C++ solvers.

    Args:
        basis: BasisSet.
        sym: Symmetry.
        occupation: Occupied orbitals per irrep (RHF/ROHF closed shells, UHF alpha),
            or None for aufbau.
        freeze_occupation: Keep the aufbau occupation after this iteration (0: never).
        skeleton: Symmetry-unique integrals only in the Fock build.
        occupation_beta: UHF beta orbitals per irrep, or None.
        occupation_open: ROHF open shells per irrep, or None.

    Returns:
        ScfSymmetry, or None for C1.

    Raises:
        ValueError: Occupation with the wrong number of irreps, or given in C1.
    """
    from ._core import ScfSymmetry
    if sym.group == "C1":
        for occ in (occupation, occupation_beta, occupation_open):
            if occ is not None and list(occ) != [sum(occ)]:
                raise ValueError("occupation per irrep requires symmetry")
        return None
    s = ScfSymmetry()
    s.group = sym.group
    s.irreps = [name for name, _ in sym.irreps]
    s.so = so_basis(basis, sym)
    for attr, occupation_in in (("occupation", occupation), ("occupation_beta", occupation_beta),
                                ("occupation_open", occupation_open)):
        if occupation_in is None:
            continue
        occ = [int(k) for k in occupation_in]
        if len(occ) != len(sym.irreps):
            raise ValueError(f"occupation needs {len(sym.irreps)} numbers "
                             f"({' '.join(s.irreps)}) for {sym.group}")
        setattr(s, attr, occ)
    s.freeze_occupation = int(freeze_occupation)
    if skeleton:
        reps = ao_operations(basis, sym)
        s.ao_perm = [[int(i) for i in p] for p, _ in reps]
        s.ao_sign = [[float(x) for x in sg] for _, sg in reps]
    s.characters = character_table(sym)
    return s


def orbital_irreps(C, eps, S, basis, sym, degeneracy=1e-5):
    """Irrep of each orbital by projection.

    Degenerate sets (equal `eps` within `degeneracy`), which may mix irreps of the
    computational group, are resolved with projectors.

    Args:
        C: Orbitals (nbf, norb).
        eps: Orbital energies (or other values defining degenerate sets).
        S: Overlap matrix.
        basis: BasisSet.
        sym: Symmetry.
        degeneracy: Tolerance for degenerate values.

    Returns:
        (irrep index (0-based) of each orbital, max. deviation from a pure irrep).
    """
    reps = ao_operations(basis, sym)
    chars = np.array([[character(op, par) for op in sym.ops] for _, par in sym.irreps])
    SC = S @ C
    # characters <psi_i| R |psi_i> for all orbitals and operations
    diag = np.array([np.einsum("mi,mi->i", SC, _apply(r, C)) for r in reps]).T
    nmo = C.shape[1]
    irreps = np.full(nmo, -1)
    worst = 0.0
    i = 0
    while i < nmo:
        j = i + 1
        while j < nmo and eps[j] - eps[j - 1] < degeneracy:
            j += 1
        counts = chars @ diag[i:j].sum(axis=0) / sym.order  # irrep content of the set
        n = np.rint(counts).astype(int)
        worst = max(worst, float(np.abs(counts - n).max()))
        labels = [k for k, c in enumerate(n) for _ in range(max(c, 0))]
        if len(labels) == j - i:
            if j - i == 1:
                irreps[i] = labels[0]
            else:  # assign within the set by largest projection
                proj = diag[i:j] @ chars.T / sym.order
                for k in labels:
                    free = [m for m in range(j - i) if irreps[i + m] < 0]
                    m = max(free, key=lambda m: proj[m, k])
                    irreps[i + m] = k
        i = j
    return irreps, worst


def orbital_labels(irreps):
    """Labels 'n.k' (n-th orbital of irrep k, k from 1) of orbitals with 0-based irreps."""
    count, labels = {}, []
    for k in irreps:
        if k < 0:
            labels.append("?")
            continue
        count[k] = count.get(k, 0) + 1
        labels.append(f"{count[k]}.{k + 1}")
    return labels
