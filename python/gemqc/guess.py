"""Initial guesses for SCF: superposition of atomic densities (SAD).

Atomic densities are built in a minimal ANO basis (default ANO-R0), whose
contractions are the atomic natural orbitals 1s, 2s, ..., 2p, ... Occupations
are those of the neutral atom's ground configuration (aufbau), spherically
averaged over partially filled shells. The molecular density is then
projected onto the target basis: D = S^-1 S_TA D_A S_AT S^-1.
"""

import numpy as np

from . import _core
from .basislib import build_basis

# aufbau (Madelung) order of subshells (n, l)
_MADELUNG = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0),
             (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2),
             (7, 1)]
# ground-state configurations deviating from Madelung: Z -> {(n, l): electrons}
_EXCEPTIONS = {
    24: {(4, 0): 1, (3, 2): 5}, 29: {(4, 0): 1, (3, 2): 10},
    41: {(5, 0): 1, (4, 2): 4}, 42: {(5, 0): 1, (4, 2): 5},
    44: {(5, 0): 1, (4, 2): 7}, 45: {(5, 0): 1, (4, 2): 8},
    46: {(5, 0): 0, (4, 2): 10}, 47: {(5, 0): 1, (4, 2): 10},
    57: {(4, 3): 0, (5, 2): 1}, 58: {(4, 3): 1, (5, 2): 1},
    64: {(4, 3): 7, (5, 2): 1}, 78: {(6, 0): 1, (5, 2): 9},
    79: {(6, 0): 1, (5, 2): 10},
}


def atomic_configuration(z: int) -> dict:
    """Ground-state configuration of the neutral atom.

    Aufbau (Madelung) order with the usual exceptions (e.g. Cr, Cu).

    Returns:
        {(n, l): number of electrons}.
    """
    conf, left = {}, z
    for n, l in _MADELUNG:
        if left == 0:
            break
        conf[(n, l)] = min(left, 2 * (2 * l + 1))
        left -= conf[(n, l)]
    conf.update(_EXCEPTIONS.get(z, {}))
    return {k: v for k, v in conf.items() if v > 0}


def _inv_sqrt(s):
    w, u = np.linalg.eigh(s)
    return u @ np.diag(w ** -0.5) @ u.T


def _pseudo_inverse(s, lindep=1e-7):
    w, u = np.linalg.eigh(s)
    keep = w > lindep
    return (u[:, keep] / w[keep]) @ u[:, keep].T


def ano_density(mol, ano_basis):
    """SAD density in the minimal ANO basis (block-diagonal per atom).

    Neutral atoms in the ground configuration, spherically averaged over partially
    filled shells.

    Raises:
        ValueError: The ANO basis lacks an occupied shell.
    """
    n = ano_basis.nbf
    D = np.zeros((n, n))
    S = _core.overlap(ano_basis)
    offsets, shell_atom = ano_basis.shell_offsets, ano_basis.shell_to_atom
    for ia, z in enumerate(mol.atomic_numbers):
        conf = atomic_configuration(z)
        idx, occ, count = [], [], {}
        for ish in range(ano_basis.nshells):
            if shell_atom[ish] != ia:
                continue
            sh = ano_basis.shell(ish)
            l, nfunc = sh["l"], 2 * sh["l"] + 1 if sh["pure"] else (sh["l"] + 1) * (sh["l"] + 2) // 2
            nq = l + 1 + count.get(l, 0)  # contraction i of l <-> principal number l+1+i
            count[l] = count.get(l, 0) + 1
            idx += range(offsets[ish], offsets[ish] + nfunc)
            occ += [conf.pop((nq, l), 0) / nfunc] * nfunc
        if conf:  # occupied shells missing in the ANO basis
            raise ValueError(f"guess basis '{ano_basis.name}' lacks shells {sorted(conf)} "
                             f"for {_core.element_symbol(z)}")
        # orthonormalize the atom's contractions (ANOs are ~orthonormal already)
        X = _inv_sqrt(S[np.ix_(idx, idx)])
        D[np.ix_(idx, idx)] = X @ np.diag(occ) @ X.T
    return D


def sad_guess(mol, basis, guess_basis="ANO-R0", out=None):
    """Superposition of atomic densities, projected onto `basis`.

    D = S^-1 S_TA D_A S_AT S^-1 with the SAD density D_A in the guess basis.

    Args:
        mol: Molecule.
        basis: Target BasisSet.
        guess_basis: Minimal ANO basis of the atomic densities.
        out: Output printer for a summary line, or None.

    Returns:
        Total density matrix (nbf, nbf).
    """
    ano = build_basis(mol, guess_basis)
    D_A = ano_density(mol, ano)
    S = _core.overlap(basis)
    Sinv = _pseudo_inverse(S)
    S_TA = _core.overlap(basis, ano)
    D = Sinv @ S_TA @ D_A @ S_TA.T @ Sinv
    if out:
        nel = sum(mol.atomic_numbers)
        out(f"  SAD guess ({guess_basis}): {np.sum(D * S):.4f} of {nel} electrons "
            f"(neutral atoms) represented in {basis.name}")
    return D

