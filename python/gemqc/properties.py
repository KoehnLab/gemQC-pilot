"""Molecular properties: dipole and quadrupole moments.

Moments refer to the centre of mass (standard atomic weights). The quadrupole moment
is the traceless (Buckingham) one, Theta_ab = 1/2 sum_i q_i (3 r_ia r_ib - r_i^2 delta_ab),
components in the order xx, yy, zz, xy, xz, yz (Molpro). All values in atomic units;
the dipole norm also in Debye.
"""

import numpy as np

from . import _core
from .elements import atomic_mass
from .units import AU_DIPOLE_IN_DEBYE

QUADRUPOLE_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))  # of QUADRUPOLE_LABELS
_SECOND = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}  # second_moments


def center_of_mass(mol):
    """Centre of mass (bohr) with standard atomic weights."""
    m = np.array([atomic_mass(z) for z in mol.atomic_numbers])
    return m @ np.asarray(mol.coordinates) / m.sum()


def buckingham(Q):
    """Traceless quadrupole (xx, yy, zz, xy, xz, yz) from second moments Q (3, 3)."""
    T = 0.5 * (3.0 * Q - np.trace(Q) * np.eye(3))
    return np.array([T[a, b] for a, b in _PAIRS])


class Multipoles:
    """Dipole and second-moment integrals about the centre of mass.

    Args:
        mol: Molecule.
        basis: BasisSet.

    Attributes:
        origin: Centre of mass (bohr).
        dipole_nuc: Nuclear dipole (3,).
        quadrupole_nuc: Nuclear traceless quadrupole (6,).
    """

    def __init__(self, mol, basis):
        self.origin = center_of_mass(mol)
        o = [float(x) for x in self.origin]
        self.X = [np.asarray(x) for x in _core.dipole(basis, o)]
        self.Q = [np.asarray(x) for x in _core.second_moments(basis, o)]
        r = np.asarray(mol.coordinates) - self.origin
        z = np.asarray(mol.atomic_numbers, float)
        self.dipole_nuc = z @ r
        self.quadrupole_nuc = buckingham(np.einsum("i,ia,ib->ab", z, r, r))

    def dipole_el(self, D):
        """Electronic dipole -tr(D r) of an AO (transition) density."""
        return -np.array([np.sum(D * x) for x in self.X])

    def quadrupole_el(self, D):
        """Electronic traceless quadrupole of an AO density."""
        Q = np.zeros((3, 3))
        for (a, b), k in _SECOND.items():
            Q[a, b] = Q[b, a] = -np.sum(D * self.Q[k])
        return buckingham(Q)

    def moments(self, D):
        """{dipole, quadrupole (totals), *_el, *_nuc} of an AO density."""
        d, q = self.dipole_el(D), self.quadrupole_el(D)
        return {"dipole": d + self.dipole_nuc, "dipole_el": d, "dipole_nuc": self.dipole_nuc,
                "quadrupole": q + self.quadrupole_nuc, "quadrupole_el": q,
                "quadrupole_nuc": self.quadrupole_nuc}


def _clean(v):
    """Values with round-off zeros set to 0 (no -0.000 in the output)."""
    v = np.asarray(v, float)
    return np.where(np.abs(v) < 1e-12, 0.0, v)


def origin_line(mp):
    """Header line naming the origin of the moments."""
    return ("  Properties   (origin of the moments: centre of mass, "
            + " ".join(f"{x:.6f}" for x in _clean(mp.origin)) + " bohr)")


def print_moments(mp, m, out):
    """Dipole and quadrupole of one wavefunction: electronic, nuclear, total."""
    out()
    out(origin_line(mp))
    out(f"  {'dipole moment / au':<22s}" + "".join(f"{c:>16s}" for c in "xyz")
        + f"{'|mu| / Debye':>16s}")
    for lab, key in (("electronic", "dipole_el"), ("nuclear", "dipole_nuc"), ("total", "dipole")):
        v = _clean(m[key])
        norm = f"{np.linalg.norm(v) * AU_DIPOLE_IN_DEBYE:16.6f}" if key == "dipole" else ""
        out(f"    {lab:<20s}" + "".join(f"{x:16.9f}" for x in v) + norm)
    out("  quadrupole moment / au (traceless, Buckingham)")
    out(f"  {'':<22s}" + "".join(f"{c:>16s}" for c in QUADRUPOLE_LABELS))
    for lab, key in (("electronic", "quadrupole_el"), ("nuclear", "quadrupole_nuc"),
                     ("total", "quadrupole")):
        out(f"    {lab:<20s}" + "".join(f"{x:16.9f}" for x in _clean(m[key])))


def moment_results(mp, m):
    """JSON entries of the moments of one wavefunction."""
    return {"properties_origin": mp.origin, "dipole": m["dipole"],
            "dipole_debye": float(np.linalg.norm(m["dipole"]) * AU_DIPOLE_IN_DEBYE),
            "quadrupole": m["quadrupole"]}
