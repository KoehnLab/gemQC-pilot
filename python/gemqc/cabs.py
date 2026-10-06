"""Complementary auxiliary basis set (CABS) for F12 methods: the RI basis, CABS orbitals,
Fock-type matrices in the orbital + CABS space and the CABS singles correction.

- RI basis: the shells of the orbital basis and of the CABS (auxiliary) basis on each
  atom, as one BasisSet (orbital-basis shells first on each atom).
- CABS orbitals (CABS+): the auxiliary functions with the orbital basis projected out,
  ``(1 - sum_p |p><p|) |a>``, canonically orthonormalized; eigenvalues of their overlap below
  a threshold are dropped (as Molpro's "Construction of CABS").
- Fock matrices (closed shell): f = h + 2 J - K of the SCF density, computed in the RI
  basis (exact or density-fitted J/K) and transformed to the orbitals (all orbital-basis
  MOs p, then the CABS orbitals x); h + 2J and K are kept for the F12 intermediates.
- CABS singles: second-order singles from the occupied orbitals i into the virtual and
  CABS orbitals A with this Fock matrix. Both blocks are made semicanonical, so the
  equations decouple: ``E = -2 sum_iA |f_iA|^2 / (e_A - e_i)``.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import _core
from .basislib import atom_shells, build_basis


def _nfunc(shells):
    return sum(2 * s.l + 1 if s.pure else (s.l + 1) * (s.l + 2) // 2 for s in shells)


def ri_basis(mol, basis_name, cabs_name):
    """The RI basis (orbital and CABS shells per atom) and the positions of the
    orbital-basis and auxiliary functions in it.

    Returns:
        (BasisSet, obs index array, auxiliary index array).
    """
    build_basis(mol, cabs_name)  # checks (ECPs, missing elements)
    zs = list(mol.atomic_numbers)
    obs, aux = atom_shells(basis_name, zs), atom_shells(cabs_name, zs)
    ri = _core.BasisSet(mol, [o + a for o, a in zip(obs, aux)], f"{basis_name}+{cabs_name}")
    obs_idx, aux_idx, pos = [], [], 0
    for o, a in zip(obs, aux):  # per atom: orbital shells, then auxiliary shells
        n_o, n_a = _nfunc(o), _nfunc(a)
        obs_idx.extend(range(pos, pos + n_o))
        aux_idx.extend(range(pos + n_o, pos + n_o + n_a))
        pos += n_o + n_a
    if pos != ri.nbf:
        raise RuntimeError("RI basis: unexpected number of functions")
    return ri, np.array(obs_idx, int), np.array(aux_idx, int)


def cabs_orbitals(S, C_obs, obs_idx, aux_idx, threshold=1e-8):
    """CABS orbitals in the RI basis.

    Args:
        S: Overlap of the RI basis.
        C_obs: Orbital-basis MOs (nobs_ao, nmo), orthonormal.
        obs_idx, aux_idx: Positions of the orbital and auxiliary functions in the RI basis.
        threshold: Eigenvalues of the projected overlap below it are dropped.

    Returns:
        (C_cabs (nri_ao, ncabs), smallest kept eigenvalue, smallest eigenvalue of the
        auxiliary overlap, number of dropped functions).
    """
    naux = len(aux_idx)
    P = np.zeros((S.shape[0], naux))  # |a> - sum_p |p><p|a> in the RI basis
    P[aux_idx, np.arange(naux)] = 1.0
    P[obs_idx, :] = -C_obs @ (C_obs.T @ S[np.ix_(obs_idx, aux_idx)])
    Sp = P.T @ S @ P
    w, U = np.linalg.eigh(0.5 * (Sp + Sp.T))
    keep = w > threshold
    eig_aux = np.linalg.eigvalsh(S[np.ix_(aux_idx, aux_idx)])[0]
    Cc = P @ (U[:, keep] / np.sqrt(w[keep]))
    return Cc, (w[keep].min() if keep.any() else 0.0), eig_aux, int((~keep).sum())


@dataclass
class RISpace:
    """Orbital-basis MOs and CABS orbitals with Fock-type matrices in this space
    (closed-shell reference).

    Attributes:
        basis: RI basis (BasisSet).
        name: CABS basis name.
        C: Orbitals in the RI basis (nri_ao, nmo + ncabs): orbital-basis MOs, then CABS.
        nmo: Number of orbital-basis MOs.
        ncabs: Number of CABS orbitals.
        occupied: Indices of the occupied MOs.
        fock: f = h + 2J - K in the orbitals C.
        hj: h + 2J in the orbitals C.
        exchange: K in the orbitals C.
        eig_kept: Smallest kept eigenvalue of the projected CABS overlap.
        eig_aux: Smallest eigenvalue of the auxiliary-basis overlap.
        dropped: CABS functions dropped (linear dependencies).
        jk_auxbasis: Fitting basis of the Fock matrices (None: exact integrals).
        time: Setup time in seconds.
        obs_idx, aux_idx: Positions of the orbital-basis and auxiliary functions in the RI
            basis.
        obs_basis: The orbital basis (BasisSet; AO order = obs_idx).
    """
    basis: object
    name: str
    C: np.ndarray
    nmo: int
    ncabs: int
    occupied: np.ndarray
    fock: np.ndarray
    hj: np.ndarray
    exchange: np.ndarray
    eig_kept: float
    eig_aux: float
    dropped: int
    jk_auxbasis: str
    time: float
    obs_idx: np.ndarray
    aux_idx: np.ndarray
    obs_basis: object = None

    @property
    def virtual(self):
        """Indices of the virtual MOs and the CABS orbitals."""
        occ = set(self.occupied.tolist())
        return np.array([p for p in range(self.nmo) if p not in occ]
                        + list(range(self.nmo, self.nmo + self.ncabs)), int)


def build_ri_space(mol, basis_name, C_obs, nocc, cabs_name, threshold=1e-8,
                   jk_auxbasis=None, screening=1e-12):
    """RISpace of a closed-shell reference.

    Args:
        mol: Molecule.
        basis_name: Orbital basis name.
        C_obs: Orbital-basis MOs (nbf, nmo), the nocc occupied ones first (e.g. the
            orbitals of correlation.orbital_spaces: frozen, active occupied, virtual,
            deleted).
        nocc: Number of (doubly) occupied orbitals.
        cabs_name: CABS (RI) basis name.
        threshold: Linear-dependency threshold of the CABS.
        jk_auxbasis: Fitting basis of density-fitted Fock matrices, or None (exact J/K).
        screening: Integral screening of exact J/K.
    """
    t0 = time.time()
    ri, obs_idx, aux_idx = ri_basis(mol, basis_name, cabs_name)
    S = np.asarray(_core.overlap(ri))
    C_obs = np.asarray(C_obs)
    Cc, eig_kept, eig_aux, dropped = cabs_orbitals(S, C_obs, obs_idx, aux_idx, threshold)
    nmo = C_obs.shape[1]
    C = np.zeros((ri.nbf, nmo + Cc.shape[1]))
    C[obs_idx, :nmo] = C_obs
    C[:, nmo:] = Cc
    Co = C[:, :nocc]
    h = np.asarray(_core.kinetic(ri)) + np.asarray(_core.nuclear_attraction(ri, mol))
    if jk_auxbasis is not None:
        builder = _core.DFBuilder(ri, build_basis(mol, jk_auxbasis, auxiliary=True))
    else:
        builder = _core.FockBuilder(ri, screening)
    (J,), (K,) = builder.compute([Co @ Co.T], True, True)
    hj = C.T @ (h + 2.0 * np.asarray(J)) @ C
    Kmo = C.T @ np.asarray(K) @ C
    return RISpace(ri, cabs_name, C, nmo, Cc.shape[1], np.arange(nocc), hj - Kmo, hj, Kmo,
                   eig_kept, eig_aux, dropped, jk_auxbasis, time.time() - t0, obs_idx, aux_idx,
                   build_basis(mol, basis_name))


def cabs_singles(ri, occ_idx=None):
    """CABS singles correction of a closed-shell reference.

    Args:
        ri: RISpace.
        occ_idx: Occupied orbitals included (default: all occupied; without the frozen
            core when the core is not correlated).

    Returns:
        (energy, contributions of the semicanonical occupied orbitals).
    """
    F = ri.fock
    occ = ri.occupied if occ_idx is None else np.asarray(occ_idx, int)
    vir = ri.virtual
    eo, Uo = np.linalg.eigh(F[np.ix_(occ, occ)])
    ev, Uv = np.linalg.eigh(F[np.ix_(vir, vir)])
    fov = Uo.T @ F[np.ix_(occ, vir)] @ Uv
    e_i = -2.0 * np.sum(fov ** 2 / (ev[None, :] - eo[:, None]), axis=1)
    return float(e_i.sum()), e_i
