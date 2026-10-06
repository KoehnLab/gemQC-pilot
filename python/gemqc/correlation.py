"""Orbital spaces of correlated methods: frozen core, deleted virtuals, semicanonical
orbitals (shared by MP2 and later correlation tasks).

Per spin the reference orbitals are split into frozen (occupied, not correlated),
occupied, virtual and deleted (virtual, not used). The correlated occupied orbitals are
classified as inner core, outer core or valence (core_shells: counts per atom; the
occupied orbitals ranked by orbital energy, as in Molpro). RHF and ROHF references share
the spatial orbitals of both spins; ROHF orbitals are not canonical for the spin Fock
matrices F^s = h + J[D] - K[D^s], so they are semicanonicalized: the occupied and
virtual blocks of each spin (alpha: closed + open / empty; beta: closed / open + empty)
are diagonalized per irrep, either only the active parts (freeze_first: frozen and
deleted orbitals selected in the reference orbitals) or the whole blocks before the
selection (diag_first). RHF and UHF orbitals are canonical (no change).
"""

from dataclasses import dataclass

import numpy as np

from . import _core
from .options import InputError


def frozen_core_orbitals(z):
    """Default number of frozen core orbitals of element z: the noble-gas core of the
    previous row; for p-block elements from row 4 on also the filled (n-1)d shell (and
    from row 6 on the 4f shell)."""
    table = ((2, 0), (10, 1), (18, 5), (30, 9), (36, 14), (48, 18), (54, 23), (57, 27),
             (71, 27), (80, 34), (86, 39), (118, 43))
    for zmax, n in table:
        if z <= zmax:
            return n
    raise ValueError(f"no frozen-core default for Z = {z}")


def core_shells(z):
    """Inner and outer core orbitals of element z: the core of frozen_core_orbitals as
    filled atomic subshells in (n, l) order (partly filled ones skipped); the outer core
    are the core subshells of the highest n, e.g. Cu: 1s2s2p (5) and 3s3p (4).

    Returns:
        (inner, outer) numbers of orbitals.
    """
    from .guess import atomic_configuration
    left, core = frozen_core_orbitals(z), []
    for (n, l), e in sorted(atomic_configuration(z).items()):
        if left > 0 and e == 2 * (2 * l + 1) and 2 * l + 1 <= left:
            core.append((n, l))
            left -= 2 * l + 1
    nmax = max((n for n, _ in core), default=0)
    outer = sum(2 * l + 1 for n, l in core if n == nmax)
    return frozen_core_orbitals(z) - left - outer, outer


OCC_CLASSES = ("inner core", "outer core", "valence")


@dataclass
class SpinSpace:
    """Orbitals of one spin in the order frozen, active occupied, active virtual, deleted.

    Attributes:
        C: Orbitals (nbf, nmo).
        eps: Orbital energies (diagonal of the spin Fock matrix; semicanonical where
            diagonalized).
        irreps: Irrep (0-based) of each orbital.
        nfrozen, nocc, nvirt, ndeleted: Sizes of the four blocks.
        F: Spin Fock matrix in these orbitals (nmo, nmo), or None for canonical references.
        occ_class: Class of each correlated occupied orbital (index into OCC_CLASSES:
            inner core, outer core, valence), or None.
    """
    C: np.ndarray
    eps: np.ndarray
    irreps: np.ndarray
    nfrozen: int
    nocc: int
    nvirt: int
    ndeleted: int
    F: np.ndarray = None
    occ_class: np.ndarray = None

    @property
    def frozen(self):
        return slice(0, self.nfrozen)

    @property
    def occ(self):
        """Active occupied orbitals."""
        return slice(self.nfrozen, self.nfrozen + self.nocc)

    @property
    def virt(self):
        """Active virtual orbitals."""
        a = self.nfrozen + self.nocc
        return slice(a, a + self.nvirt)

    @property
    def deleted(self):
        a = self.nfrozen + self.nocc + self.nvirt
        return slice(a, a + self.ndeleted)

    def labels(self, block=slice(None)):
        """Orbital labels n.k (n-th orbital of irrep k in the order frozen, active
        occupied, ...; for canonical orbitals the labels of the SCF) of a block."""
        from .symmetry import orbital_labels
        return orbital_labels(self.irreps)[block]

    def per_irrep(self, block, nirr):
        """Number of orbitals per irrep in a block (slice or index array)."""
        return [int(np.sum(self.irreps[block] == k)) for k in range(nirr)]

    def occ_of_class(self, c):
        """Indices of the correlated occupied orbitals of class c (see OCC_CLASSES)."""
        return np.arange(self.nfrozen, self.nfrozen + self.nocc)[self.occ_class == c]

    def occ_labels(self, flag_core=False):
        """Labels n.k of the correlated occupied orbitals; flag_core: core orbitals marked
        with '*' (others padded with a blank for alignment)."""
        lab = self.labels(self.occ)
        if not flag_core:
            return lab
        return [l + ("*" if c else " ") for l, c in zip(lab, self.core_mask())]

    def core_mask(self):
        """True for the correlated occupied orbitals of the core (inner or outer)."""
        return self.occ_class < 2

    def classify(self, n_inner, n_outer):
        """Set occ_class: all occupied orbitals (frozen included) ranked by orbital energy,
        the lowest n_inner inner core, the next n_outer outer core, the rest valence."""
        n = self.nfrozen + self.nocc
        rank = np.empty(n, int)
        rank[np.argsort(self.eps[:n], kind="stable")] = np.arange(n)
        cls = np.where(rank < n_inner, 0, np.where(rank < n_inner + n_outer, 1, 2))
        self.occ_class = cls[self.nfrozen:]


@dataclass
class OrbitalSpaces:
    """Correlated orbital spaces of a reference wavefunction.

    Attributes:
        reference: 'RHF', 'ROHF' or 'UHF'.
        alpha, beta: SpinSpace (the same object for RHF).
        semicanonical: 'freeze_first' or 'diag_first' (only relevant for ROHF).
    """
    reference: str
    alpha: SpinSpace
    beta: SpinSpace
    semicanonical: str


def _reference(wf):
    m = wf.method.upper()
    if m.startswith("RHF"):
        return "RHF"
    if m.startswith("ROHF"):
        return "ROHF"
    if m.startswith("UHF"):
        return "UHF"
    raise InputError(f"correlated methods need an RHF, ROHF or UHF reference, not {wf.method}")


def _count(spec, what, pool_eps, pool_irreps, nirr, default, lowest):
    """Selected orbitals of a pool (indices into the pool) for frozen/deleted input:
    'auto' (default count), 'off', a number (lowest/highest by energy) or a list per irrep
    (lowest/highest in each irrep)."""
    n = len(pool_eps)
    order = np.argsort(pool_eps if lowest else -pool_eps, kind="stable")
    if spec is None or spec == "off" or spec is False:
        return np.zeros(0, int)
    if spec == "auto":
        spec = default
    if isinstance(spec, (list, tuple)):
        if len(spec) != nirr:
            raise InputError(f"{what}: {nirr} numbers expected (one per irrep)")
        out = []
        for k, c in enumerate(spec):
            idx = [i for i in order if pool_irreps[i] == k]
            if not 0 <= int(c) <= len(idx):
                raise InputError(f"{what}: {c} orbitals in irrep {k + 1}, only {len(idx)} available")
            out += idx[:int(c)]
        return np.array(sorted(out), int)
    try:
        c = int(spec)
    except (TypeError, ValueError):
        raise InputError(f"{what}: expected auto, off, a number or a list per irrep, not "
                         f"{spec!r}") from None
    if not 0 <= c <= n:
        raise InputError(f"{what}: {c} orbitals requested, only {n} available")
    return np.array(sorted(order[:c]), int)


def spin_fock(wf, jk=None, threshold=1e-12):
    """AO spin Fock matrices (F_alpha, F_beta) of a ROHF (or any) wavefunction."""
    C, occ = np.asarray(wf.C), np.asarray(wf.occ)
    Da = C[:, occ > 0.5] @ C[:, occ > 0.5].T
    Db = C[:, occ > 1.5] @ C[:, occ > 1.5].T
    h = np.asarray(_core.kinetic(wf.basis)) + np.asarray(_core.nuclear_attraction(wf.basis, wf.molecule))
    builder = jk if jk is not None else _core.FockBuilder(wf.basis, threshold)
    J, K = builder.compute([Da, Db], True, True)
    Jt = np.asarray(J[0]) + np.asarray(J[1])
    return h + Jt - np.asarray(K[0]), h + Jt - np.asarray(K[1])


def _diagonalize(C, irreps, F, idx):
    """Diagonalize F (AO) in the orbitals C[:, idx] per irrep; (C block, eps block, irreps)."""
    Cb, eb, ib = [], [], []
    for k in sorted(set(irreps[idx].tolist())):
        sel = idx[irreps[idx] == k]
        Ck = C[:, sel]
        w, U = np.linalg.eigh(Ck.T @ F @ Ck)
        Cb.append(Ck @ U)
        eb.append(w)
        ib.append(np.full(len(sel), k))
    if not Cb:
        return C[:, :0], np.zeros(0), np.zeros(0, int)
    Cb, eb, ib = np.hstack(Cb), np.concatenate(eb), np.concatenate(ib)
    o = np.argsort(eb, kind="stable")
    return Cb[:, o], eb[o], ib[o]


def _spin_space(C, eps, irreps, frozen, occ, virt, deleted, F=None):
    """SpinSpace from index arrays into (C, eps, irreps) of the four blocks."""
    parts = [frozen, occ, virt, deleted]
    order = np.concatenate(parts).astype(int)
    Cs = C[:, order]
    return SpinSpace(Cs, eps[order], irreps[order], len(frozen), len(occ), len(virt),
                     len(deleted), None if F is None else Cs.T @ F @ Cs)


def orbital_spaces(wf, frozen="auto", deleted="off", semicanonical="freeze_first", jk=None):
    """Frozen, active and deleted orbitals of a reference wavefunction.

    Args:
        wf: SCF wavefunction (RHF, ROHF or UHF).
        frozen: 'auto' (frozen_core_orbitals per atom), 'off', a number of orbitals (the
            lowest occupied by energy) or a list per irrep.
        deleted: 'off', a number (the highest virtuals by energy) or a list per irrep.
        semicanonical: 'freeze_first' or 'diag_first' (ROHF references).
        jk: J/K builder for the spin Fock matrices of ROHF (default: exact integrals).

    Returns:
        OrbitalSpaces (occupied orbitals classified, SpinSpace.occ_class).

    Raises:
        InputError: Invalid counts or reference.
    """
    spaces = _orbital_spaces(wf, frozen, deleted, semicanonical, jk)
    n_inner, n_outer = (sum(x) for x in zip(*(core_shells(int(z))
                                               for z in wf.molecule.atomic_numbers)))
    for s in {id(spaces.alpha): spaces.alpha, id(spaces.beta): spaces.beta}.values():
        s.classify(n_inner, n_outer)
    return spaces


def _orbital_spaces(wf, frozen, deleted, semicanonical, jk):
    """orbital_spaces without the classification of the occupied orbitals."""
    if semicanonical not in ("freeze_first", "diag_first"):
        raise InputError(f"semicanonical: freeze_first or diag_first, not {semicanonical!r}")
    ref = _reference(wf)
    nirr = max(1, len(wf.irrep_names))
    n_core = sum(frozen_core_orbitals(int(z)) for z in wf.molecule.atomic_numbers)
    if ref == "UHF":
        spins = []
        for C, eps, occ, irr in ((wf.C, wf.eps, wf.occ, wf.irreps),
                                 (wf.C_b, wf.eps_b, wf.occ_b, wf.irreps_b)):
            C, eps, occ, irr = (np.asarray(x) for x in (C, eps, occ, irr))
            o, v = np.nonzero(occ > 0.5)[0], np.nonzero(occ <= 0.5)[0]
            fr = o[_count(frozen, "frozen", eps[o], irr[o], nirr, n_core, True)]
            de = v[_count(deleted, "deleted", eps[v], irr[v], nirr, 0, False)]
            spins.append(_spin_space(C, eps, irr, fr, np.setdiff1d(o, fr), np.setdiff1d(v, de), de))
        return OrbitalSpaces(ref, spins[0], spins[1], semicanonical)
    C, eps, occ, irr = (np.asarray(x) for x in (wf.C, wf.eps, wf.occ, wf.irreps))
    closed, open_, empty = (np.nonzero(occ > 1.5)[0], np.nonzero((occ > 0.5) & (occ <= 1.5))[0],
                            np.nonzero(occ <= 0.5)[0])
    if ref == "RHF":
        fr = closed[_count(frozen, "frozen", eps[closed], irr[closed], nirr, n_core, True)]
        de = empty[_count(deleted, "deleted", eps[empty], irr[empty], nirr, 0, False)]
        s = _spin_space(C, eps, irr, fr, np.setdiff1d(closed, fr), np.setdiff1d(empty, de), de)
        return OrbitalSpaces(ref, s, s, semicanonical)
    # ROHF: spin Fock matrices, semicanonical orbitals per spin
    Fa, Fb = spin_fock(wf, jk)
    blocks = {"a": (np.concatenate([closed, open_]), empty, Fa),
              "b": (closed, np.concatenate([open_, empty]), Fb)}
    spins = []
    if semicanonical == "freeze_first":
        # frozen among the closed shells and deleted among the empty orbitals (reference
        # orbital energies), the same spatial orbitals for both spins
        fr = closed[_count(frozen, "frozen", eps[closed], irr[closed], nirr, n_core, True)]
        de = empty[_count(deleted, "deleted", eps[empty], irr[empty], nirr, 0, False)]
        for key in "ab":
            o, v, F = blocks[key]
            Co, eo, io = _diagonalize(C, irr, F, np.setdiff1d(o, fr))
            Cv, ev, iv = _diagonalize(C, irr, F, np.setdiff1d(v, de))
            Cs = np.hstack([C[:, fr], Co, Cv, C[:, de]])
            es = np.concatenate([np.einsum("mi,mn,ni->i", C[:, fr], F, C[:, fr]), eo, ev,
                                 np.einsum("mi,mn,ni->i", C[:, de], F, C[:, de])])
            ir = np.concatenate([irr[fr], io, iv, irr[de]])
            spins.append(SpinSpace(Cs, es, ir, len(fr), len(eo), len(ev), len(de), Cs.T @ F @ Cs))
    else:  # diag_first: whole occupied / virtual blocks, then selection per spin
        for key in "ab":
            o, v, F = blocks[key]
            Co, eo, io = _diagonalize(C, irr, F, o)
            Cv, ev, iv = _diagonalize(C, irr, F, v)
            fr = _count(frozen, "frozen", eo, io, nirr, n_core, True)
            de = _count(deleted, "deleted", ev, iv, nirr, 0, False)
            if len(fr) > len(closed):
                raise InputError(f"frozen: {len(fr)} orbitals, only {len(closed)} closed shells")
            ao, av = np.setdiff1d(np.arange(len(eo)), fr), np.setdiff1d(np.arange(len(ev)), de)
            Cs = np.hstack([Co[:, fr], Co[:, ao], Cv[:, av], Cv[:, de]])
            es = np.concatenate([eo[fr], eo[ao], ev[av], ev[de]])
            ir = np.concatenate([io[fr], io[ao], iv[av], iv[de]])
            spins.append(SpinSpace(Cs, es, ir, len(fr), len(ao), len(av), len(de), Cs.T @ F @ Cs))
    return OrbitalSpaces(ref, spins[0], spins[1], semicanonical)


PARTS = ("core", "core-valence", "valence")


def pair_partition(e, core_i, core_j):
    """Core, core-valence and valence parts of pair energies e[i, j] (summed over the pairs
    of both, one or no core orbitals).

    Args:
        e: Pair energies (n_i, n_j); zero entries for pairs not stored (e.g. i < j).
        core_i, core_j: Core flags of the orbitals i and j (SpinSpace.core_mask).

    Returns:
        np.ndarray (core, core-valence, valence).
    """
    ci, cj = np.asarray(core_i)[:, None], np.asarray(core_j)[None, :]
    return np.array([np.sum(e * (ci & cj)), np.sum(e * (ci ^ cj)), np.sum(e * (~ci & ~cj))])


def has_correlated_core(spaces):
    """Whether core orbitals (inner or outer) are correlated."""
    return bool(spaces.alpha.core_mask().any() or spaces.beta.core_mask().any())


def space_results(spaces, nirr):
    """JSON entries of the orbital spaces (alpha): frozen_per_irrep, occupied_per_irrep
    {inner_core, outer_core, valence}, virtual_per_irrep, deleted_per_irrep."""
    a, nirr = spaces.alpha, max(1, nirr)
    return {"frozen_per_irrep": a.per_irrep(a.frozen, nirr),
            "occupied_per_irrep": {name.replace(" ", "_"): a.per_irrep(a.occ_of_class(c), nirr)
                                   for c, name in enumerate(OCC_CLASSES)},
            "virtual_per_irrep": a.per_irrep(a.virt, nirr),
            "deleted_per_irrep": a.per_irrep(a.deleted, nirr)}


def print_spaces(spaces, irrep_names, out):
    """Frozen, occupied (inner core, outer core, valence), virtual and deleted orbitals
    per irrep and in total (and spin); core rows only if not empty."""
    nirr = max(1, len(irrep_names))
    names = irrep_names or ["A"]
    out.value("reference", spaces.reference + (f", semicanonical orbitals ({spaces.semicanonical})"
                                               if spaces.reference == "ROHF" else ""))
    out(f"  {'orbitals per irrep':<34s}" + "".join(f"{n:>6s}" for n in names) + f"{'total':>8s}")
    spin_list = [("", spaces.alpha)] if spaces.alpha is spaces.beta else [
        (" (alpha)", spaces.alpha), (" (beta)", spaces.beta)]
    for tag, s in spin_list:
        rows = [("frozen", s.frozen)]
        rows += [(f"occupied ({name})", s.occ_of_class(c)) for c, name in enumerate(OCC_CLASSES)]
        rows += [("virtual", s.virt), ("deleted", s.deleted)]
        for label, blk in rows:
            n = s.per_irrep(blk, nirr)
            if label.startswith("occupied") and "core" in label and sum(n) == 0:
                continue
            out(f"    {label + tag:<32s}" + "".join(f"{x:6d}" for x in n) + f"{sum(n):8d}")
