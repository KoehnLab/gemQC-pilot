"""F12XG pilot: MP2-F12 with extended geminals bound to basis-function groups
(plan: tmp_plans/IMPLEMENTATION.md).

Basis-function groups: each AO of the orbital basis gets a group label G (0-based); the
geminal of the AO pair (mu, nu) is f_GH with gamma_GH = (gt_G + gt_H)/2
(geminal.XGGeminal).

Automatic core/valence groups (group 1 core, group 2 valence):

- core orbital functions: the second moment of each normalised AO about its own atom,
  <r_A^2>, below threshold * r2_core (HF <r^2> of the outermost core shell of the element;
  default 1.05: the HF-contracted core functions of cc-pVXZ have ratios <= 1);
- core-correlating functions: all functions of the orbital basis that are not in a valence
  reference basis (cc-pCVXZ-F12 -> cc-pVXZ-F12, derived per atom from the name or given). These
  cannot be found from <r^2>: e.g. the extra p of C in cc-pCVDZ-F12 has the ratio 3.2,
  the inner 2p function of cc-pVDZ-F12 2.8 (Al: extra p 2.4, valence d 1.9).

Segmented basis sets (def2) split the core orbitals into several functions spread between
core and valence (ratios 0.3-4, no gap), so there the core/valence split is a convention;
such sets may not be suited for F12XG.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import _core

#: HF <r^2> (bohr^2) of the outermost core shell (Li-Ne: 1s, Na-Ar: max of 2s, 2p,
#: Ga-Kr: max of 3s, 3p, 3d = 3d), from the HF-contracted core functions of cc-pVQZ
#: (Li-Ar) or atomic ROHF in cc-pVQZ (Ga-Kr; reproduces the Li-Ar values)
CORE_R2 = {3: 0.4468, 4: 0.2330, 5: 0.1434, 6: 0.0972, 7: 0.0703, 8: 0.0531, 9: 0.0416,
           10: 0.0335, 11: 0.8222, 12: 0.5977, 13: 0.4589, 14: 0.3773, 15: 0.3159,
           16: 0.2685, 17: 0.2312, 18: 0.2012,
           31: 0.7962, 32: 0.6571, 33: 0.5566, 34: 0.4800, 35: 0.4199, 36: 0.3715}

#: default of core_valence_threshold (ratio <r_A^2> / r2_core of the core orbital functions)
CORE_VALENCE_THRESHOLD = 1.05


def check_core_reference(z):
    """Raise if element z has a core but no entry in CORE_R2."""
    if z > 2 and z not in CORE_R2:
        raise ValueError(f"core_valence: no core reference for {_core.element_symbol(z)} "
                         "(H-Ar, Ga-Kr so far); give bf_groups by atoms or functions")


def valence_reference_name(name, zs):
    """Valence basis of a core-valence basis by its (canonical) name, per atom cc-pCVXZ,
    cc-pwCVXZ -> cc-pVXZ (canonical name), or None if no atom has such a basis.

    Args:
        name: Orbital basis name (plain or canonical, see basislib.canonical_name).
        zs: Atomic numbers.
    """
    from .basislib import basis_names, canonical_name
    names = basis_names(name, zs)
    ref = [n.replace("pwCV", "pV").replace("pCV", "pV") for n in names]
    return canonical_name(ref, zs) if ref != names else None


def _shell_key(sh):
    """Atom, l and contraction (rounded) of a shell, to compare basis sets."""
    return (sh["atom"], sh["l"], sh["pure"],
            tuple(float(f"{x:.10g}") for x in sh["exponents"]),
            tuple(float(f"{x:.8g}") for x in sh["coefficients"]))


#: relative residual below which a reference function counts as spanned by the basis
SPAN_TOLERANCE = 1e-8


def _spanned(basis, ref, s, tol):
    """Whether the AOs of shell s of ref lie in the span of the AOs of basis with the same
    atom and l (residual of the projection relative to the norm below tol)."""
    sh = ref.shell(s)
    l, _ = ao_shells(basis)
    B = np.nonzero((np.asarray(basis.bf_to_atom) == sh["atom"]) & (l == sh["l"]))[0]
    if B.size == 0:
        return False
    off = list(ref.shell_offsets) + [ref.nbf]
    i = np.arange(off[s], off[s + 1])
    S = np.asarray(_core.overlap(basis))[np.ix_(B, B)]
    X = np.asarray(_core.overlap(basis, ref))[np.ix_(B, i)]
    norm = np.diag(np.asarray(_core.overlap(ref)))[i]
    res = norm - np.einsum("bi,bi->i", X, np.linalg.solve(S, X))
    return bool(np.all(res <= tol * norm))


def extra_functions(basis, ref, tol=SPAN_TOLERANCE):
    """The AOs of basis in shells that ref does not have (same molecule).

    A shell of ref without an identical shell in basis is accepted if it lies in the span
    of the functions of basis on its atom with its l (recontracted sets, e.g. the p set
    of aug-cc-pwCV5Z for Br, where the 4p contraction of cc-pV5Z is dropped).

    Returns:
        ((nbf,) bool extra AOs, number of shells of ref only contained by their span).

    Raises:
        ValueError: ref has functions outside the span of basis.
    """
    left = {}
    for s in range(ref.nshells):
        left.setdefault(_shell_key(ref.shell(s)), []).append(s)
    off = list(basis.shell_offsets) + [basis.nbf]
    extra = np.zeros(basis.nbf, bool)
    for s in range(basis.nshells):
        k = _shell_key(basis.shell(s))
        if left.get(k):
            left[k].pop()
        else:
            extra[off[s]:off[s + 1]] = True
    missing = [s for v in left.values() for s in v]
    if not all(_spanned(basis, ref, s, tol) for s in missing):
        raise ValueError(f"core_valence: reference basis '{ref.name}' is not contained in "
                         f"'{basis.name}'")
    return extra, len(missing)


def ao_shells(basis):
    """Angular momentum and shell index of each AO."""
    off = list(basis.shell_offsets) + [basis.nbf]
    l, shell = np.zeros(basis.nbf, int), np.zeros(basis.nbf, int)
    for s in range(basis.nshells):
        l[off[s]:off[s + 1]] = basis.shell(s)["l"]
        shell[off[s]:off[s + 1]] = s
    return l, shell


def ao_r2(mol, basis):
    """<r_A^2> of each normalised AO about its own atom A (bohr^2):
    <r^2>_0 - 2 A.<r>_0 + A^2 with moments about the origin."""
    o = [0.0, 0.0, 0.0]
    S = np.diag(np.asarray(_core.overlap(basis)))
    r = np.array([np.diag(np.asarray(x)) for x in _core.dipole(basis, o)])        # (3, nbf)
    Q = _core.second_moments(basis, o)
    r2 = sum(np.diag(np.asarray(Q[k])) for k in (0, 3, 5))
    A = np.asarray(mol.coordinates)[np.asarray(basis.bf_to_atom)]                 # (nbf, 3)
    return (r2 - 2 * np.einsum("an,na->n", r, A)) / S + np.einsum("na,na->n", A, A)


@dataclass
class BfGroups:
    """Group label (0-based) of each AO of the orbital basis.

    Attributes:
        labels: (nbf,) group of each AO.
        ngroups: Number of groups.
        how: Description of the assignment.
        r2: <r_A^2> per AO (core_valence) or None.
    """
    labels: np.ndarray
    ngroups: int
    how: str
    r2: np.ndarray = None

    def mask(self, G):
        """(nbf,) bool: the AOs of group G."""
        return self.labels == G

    def summary(self, mol, basis):
        """Lines for the output: AOs per group and atom (and the <r^2> range per group)."""
        atoms = np.asarray(basis.bf_to_atom)
        z = mol.atomic_numbers
        lines = [f"basis-function groups: {self.how}"]
        for G in range(self.ngroups):
            m = self.mask(G)
            per = [f"{_core.element_symbol(int(z[a]))}{a + 1}:{np.sum(m & (atoms == a))}"
                   for a in range(len(z)) if np.any(m & (atoms == a))]
            line = f"group {G + 1}: {np.sum(m)} functions ({' '.join(per) or 'none'})"
            if self.r2 is not None and np.any(m):
                line += f", <r^2> {self.r2[m].min():.3g} ... {self.r2[m].max():.3g} bohr^2"
            lines.append(line)
        return lines


def assign_bf_groups(mol, basis, spec, threshold=CORE_VALENCE_THRESHOLD):
    """Basis-function groups from the option bf_groups.

    Args:
        mol: Molecule.
        basis: Orbital basis (BasisSet).
        spec: 'core_valence' (group 1 core, group 2 valence; reference basis from the
            name), {'core_valence': reference basis name}, {'atoms': [..]} (group 1:
            these atoms, 1-based numbers or element symbols; group 2: the others) or
            {'functions': [..]} (group 1 or 2 of each AO).
        threshold: core_valence: ratio <r_A^2> / r2_core below which an AO is a core
            orbital function.

    Returns:
        BfGroups (two groups).

    Raises:
        ValueError: Malformed spec.
    """
    nbf = basis.nbf
    atoms = np.asarray(basis.bf_to_atom)
    z = [int(x) for x in mol.atomic_numbers]
    cv = None
    if isinstance(spec, str) and spec.lower() == "core_valence":
        cv = valence_reference_name(basis.name, list(mol.atomic_numbers)), True
    elif isinstance(spec, dict) and set(spec) == {"core_valence"}:
        if not isinstance(spec["core_valence"], str):
            raise ValueError("bf_groups: core_valence needs a reference basis name")
        cv = spec["core_valence"], False
    if cv is not None:
        return _core_valence(mol, basis, *cv, threshold)
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError("bf_groups: core_valence, {core_valence: <basis>}, {atoms: [..]} "
                         "or {functions: [..]}")
    key, val = next(iter(spec.items()))
    if not isinstance(val, (list, tuple)):
        raise ValueError(f"bf_groups: {key} needs a list")
    if key == "atoms":
        sel = set()
        for a in val:
            if isinstance(a, str) and not a.isdigit():
                hit = [i for i, zi in enumerate(z)
                       if _core.element_symbol(zi).lower() == a.lower()]
                if not hit:
                    raise ValueError(f"bf_groups: no atom of element '{a}'")
                sel.update(hit)
            else:
                i = int(a)
                if not 1 <= i <= len(z):
                    raise ValueError(f"bf_groups: atom {i} out of range 1..{len(z)}")
                sel.add(i - 1)
        labels = np.where(np.isin(atoms, sorted(sel)), 0, 1)
        names = " ".join(f"{_core.element_symbol(z[i])}{i + 1}" for i in sorted(sel))
        return BfGroups(labels, 2, f"atoms (group 1: {names})")
    if key == "functions":
        labels = np.asarray(val)
        if labels.shape != (nbf,) or not np.all(np.isin(labels, (1, 2))):
            raise ValueError(f"bf_groups: functions needs {nbf} labels 1 or 2")
        return BfGroups(labels.astype(int) - 1, 2, "explicit functions")
    raise ValueError(f"bf_groups: unknown key '{key}' (core_valence, atoms, functions)")


def _core_valence(mol, basis, ref_name, derived, threshold):
    """Core/valence groups: core orbital functions by <r_A^2>, plus the functions not in the
    valence reference basis ref_name (None: none; derived: from the name, skipped if the
    basis set is not available)."""
    from .basislib import build_basis
    if not threshold > 0:
        raise ValueError("core_valence_threshold must be > 0")
    z = [int(x) for x in mol.atomic_numbers]
    for zi in set(z):
        check_core_reference(zi)
    atoms = np.asarray(basis.bf_to_atom)
    r2 = ao_r2(mol, basis)
    ref = np.array([CORE_R2.get(z[a], 0.0) for a in atoms])
    core = r2 < threshold * ref
    how = f"core_valence (core: <r_A^2> < {threshold:g} r2_core"
    if ref_name is not None:
        try:
            rb = build_basis(mol, ref_name)
        except (ValueError, FileNotFoundError):
            if not derived:
                raise
            rb = None
            how += f"; reference basis {ref_name} not available"
        if rb is not None:
            extra, nspan = extra_functions(basis, rb)
            core |= extra
            how += f", and {np.sum(extra)} functions not in {ref_name}"
            if nspan:
                how += f" ({nspan} of its shells recontracted)"
    return BfGroups(np.where(core, 0, 1), 2, how + ")", r2)


# --- exact F12XG pair integrals ------------------------------------------------------------

class ExactF12XGIntegrals:
    """F12XG pair integrals with exact integrals, for the active pairs (interface of
    f12.ExactF12Integrals.pair, consumed by f12.PairIntermediates).

    The geminal is a sum over group pairs with group-masked occupied orbitals,
    g_ij = sum_GH f_GH phi_i^G(1) phi_j^H(2), so every integral is a sum of standard pair
    integrals (C++ `pair_integrals` over the masked columns (i^G, j^H) of Cx, one call per
    operator and group pair; only the shells of G and H enter). Per pair i >= j (stored,
    memory/disk): ``F = <p'q'|g_ij>`` (N' x N') and
    ``G = <ij|r12^-1|p'q'>`` (as ExactF12Integrals). Small arrays over the active orbitals,
    [k, l, m, n] with the geminal pair (k, l) and the plain or other-geminal pair (m, n)::

        FG = <mn|g_kl/r12>                        (f_GH/r12)
        U  = <g_mn|grad_1 . grad_1|g_kl>          ((grad f_G'H') . (grad f_GH))
        F2 = <g_mn|g_kl>                          (f_G'H' f_GH)
        H1 = w <(hJ m^G') n^H'|f_G'H' f_GH|k^G l^H> (hJ = h + 2J on electron 1 via the RI;
                                                   w: kinetic_weight(G'H', GH))

    These come from the MO integrals over the masked orbitals (C++ `mo_integrals`, one call
    per distinct operator; permutational symmetry of the quartets).

    Args:
        ri: cabs.RISpace.
        nfrozen: Frozen core orbitals.
        nact: Active occupied orbitals.
        geminal: geminal.XGGeminal.
        groups: BfGroups over the orbital basis (AO order of ri.obs_basis).
        threshold: Integral screening.
        mode: Storage 'auto', 'memory' or 'disk'.
    """

    def __init__(self, ri, nfrozen, nact, geminal, groups, threshold=1e-12, mode="auto"):
        from .transform import DOUBLE, ThreeIndexStore
        t0 = time.time()
        self.act = act = np.arange(nfrozen, nfrozen + nact)
        self.nact, self.N = nact, ri.C.shape[1]
        N, nmo, ng = self.N, ri.nmo, groups.ngroups
        if ng != len(geminal.gammas):
            raise ValueError(f"{ng} basis-function groups but {len(geminal.gammas)} gammas")
        pairs = [(i, j) for i in range(nact) for j in range(i + 1)]
        ordered = [(i, j) for i in range(nact) for j in range(nact)]
        self.index = {p: n for n, p in enumerate(pairs)}
        npair = len(pairs)
        size = npair * 2 * N * N * DOUBLE
        self.store = ThreeIndexStore(size, mode, "f12xg_pair_integrals")
        for name in ("F", "G"):
            self.store.create(name, (npair, N, N))
        bo, bri = ri.obs_basis, ri.basis
        C = ri.C
        Cobs = ri.C[ri.obs_idx, :nmo]
        Co = Cobs[:, act]
        # masked active orbitals: column G * nact + i is i^G (orbital basis; RI AOs)
        Cx = np.hstack([Co * groups.mask(G)[:, None] for G in range(ng)])
        Cx_ri = np.zeros((bri.nbf, ng * nact))
        Cx_ri[ri.obs_idx] = Cx
        S = np.asarray(_core.overlap(bri))
        Hx_ri = C @ (ri.hj @ (C.T @ (S @ Cx_ri)))   # hJ i^G in the RI
        col = lambda G, i: G * nact + i  # noqa: E731

        def run(op, b1, b2, cols, per):
            """pair_integrals of op over the masked column pairs cols, in batches (the shells
            with zero coefficients are skipped): yields K for each column pair."""
            limit = _core.memory()
            nb = max(1, int(0.5 * limit // per)) if limit else len(cols)
            for k in range(0, len(cols), nb):
                yield from map(np.asarray, _core.pair_integrals(op, bo, b1, b2, Cx,
                                                                cols[k:k + nb], threshold))

        # F = <p'q'|g_ij> = sum_GH (i^G p'|f_GH|j^H q'): both free indices in the RI basis,
        # accumulated over the group pairs in the AO basis. For G != H one orientation over
        # all ordered pairs: (i^H p'|f|j^G q') = (j^G q'|f|i^H p') (b1 = b2)
        Fao = {p: np.zeros((bri.nbf, bri.nbf)) for p in pairs}
        for G, H in geminal.pairs:
            op = geminal.operators(G, H)["f12"]
            plist = pairs if G == H else ordered
            for (i, j), K in zip(plist, run(op, bri, bri, [(col(G, i), col(H, j))
                                                          for i, j in plist],
                                            bri.nbf ** 2 * DOUBLE)):
                if i >= j:
                    Fao[(i, j)] += K
                if G != H and i <= j:
                    Fao[(j, i)] += K.T
        for p, n in self.index.items():
            self.store.set("F", n, C.T @ Fao.pop(p) @ C)
        del Fao
        # G = <ij|r12^-1|p'q'>: no geminal (as ExactF12Integrals)
        coul = _core.TwoBodyOperator("coulomb")
        half = {p: C.T @ np.asarray(K) @ Cobs for p, K in zip(ordered, _core.pair_integrals(
            coul, bo, bri, bo, Co, ordered, threshold))}
        o = len(ri.occupied)
        for (i, j), n in self.index.items():
            G = np.zeros((N, N))
            G[:, :nmo] = half[(i, j)]
            G[:o, nmo:] = half[(j, i)][nmo:, :o].T  # <ij|o x> = <ji|x o>
            self.store.set("G", n, G)
        del half
        # small arrays over the active orbitals from the MO integrals over the masked
        # orbitals, Y[G, k, G1, m, H, l, H1, n] = (k^G m^G1|op|l^H n^H1) (C++ mo_integrals,
        # one call per distinct operator, all four indices transformed at once)
        na = nact

        def orient(T):
            return {T, T[::-1]}

        def mo(op, blocks, b2=bo, C2=Cx):
            """Y of op; only the blocks (G, G1, H, H1) are needed (the other quartets of
            group shells are skipped)"""
            Y = np.asarray(_core.mo_integrals(op, bo, Cx, b2, C2, bo, Cx, bo, Cx, threshold,
                                              sorted(blocks), na, ng))
            return Y.reshape(ng, na, -1, ng, na, ng, na, ng, na)

        def fg(T):  # all four indices masked (8-fold symmetry), summed over G1, H1
            Y = mo(geminal.operators(*T)["f12/r12"], {(G, G1, H, H1) for G, H in orient(T)
                                                      for G1 in range(ng) for H1 in range(ng)})
            return Y[:, :, 0].sum(axis=(2, 6), keepdims=True)

        CH = np.hstack([Cx_ri, Hx_ri])

        def prod(Tb, T):  # F2 and H1 from the bra orbitals [m^G1, hJ m^G1] (RI) in one call
            ops = geminal.product_operators(Tb, T)
            blocks = {(G, G1, H, H1) for Tk, Tbra in {(T, Tb), (Tb, T)}
                      for G, H in orient(Tk) for G1, H1 in orient(Tbra)}
            Yh = mo(ops["f12 x f12"], blocks, bri, CH)
            return mo(ops["grad f12 . grad f12"], blocks)[:, :, 0], Yh[:, :, 0], Yh[:, :, 1]

        self.FG, self.U, self.F2, self.H1 = _small_arrays(geminal, na, fg, prod)
        self.time = time.time() - t0

    def pair(self, i, j):
        """Pair integrals of the active pair (i, j), i >= j: F, G and the blocks over
        kl, mn in (ij, ji): f12/r12 (<mn|g_kl/r12>, geminal on the row pair), (grad f12)^2,
        F2 (<g_kl|g_mn>) and F2bar (hJ on the row pair: [F^2 bar] of A)."""
        n = self.index[(i, j)]
        out = {"F": self.store.get("F", n), "G": self.store.get("G", n)}
        out.update(_blocks(self, i, j))
        return out

    def close(self):
        self.store.close()


# --- density-fitted F12XG integrals ------------------------------------------------------

class DFF12XGIntegrals:
    """Density-fitted F12XG integrals (interface of ExactF12XGIntegrals).

    Robust fitting term by term of the group decomposition: each group pair has a fixed
    operator f_GH and the masked densities k^G p' with the fit coefficients
    ``d_{k^G} = J^-1 (P|k^G p')``::

        (k^G p'|f_GH|l^H q') = d_{k^G}^T G~_{l^H} + G~_{k^G}^T d_{l^H},
        G~_{k^G} = (P|f_GH|k^G p') - 1/2 (P|f_GH|Q) d_{k^G}

    Fit coefficients are linear in the orbitals, so those of doubly masked and
    hJ-transformed densities are contractions of d_{k^G}. Stored per active k (naux x N'):
    d of each mask, the Coulomb (P|k p') and Z_{k^G} = sum_H G~_{k^H} of f_GH, so that
    F_ij = sum_G d_{i^G}^T Z_{j^G} + Z_{i^G}^T d_{j^G}. The blocks FG, U, F2, H1 (as
    ExactF12XGIntegrals) come from small arrays over the masked active orbitals (orbital
    basis; RI basis for the hJ-transformed ones).

    Args:
        ri: cabs.RISpace.
        nfrozen: Frozen core orbitals.
        nact: Active occupied orbitals.
        fit: Fitting basis (BasisSet).
        geminal: geminal.XGGeminal.
        groups: BfGroups over the orbital basis (AO order of ri.obs_basis).
        mode: Storage 'auto', 'memory' or 'disk'.
    """

    def __init__(self, ri, nfrozen, nact, fit, geminal, groups, mode="auto"):
        from scipy.linalg import cho_factor, cho_solve
        from .transform import DOUBLE, ThreeIndexStore
        t0 = time.time()
        self.act = act = np.arange(nfrozen, nfrozen + nact)
        self.nact, self.N = nact, ri.C.shape[1]
        N, ng, na = self.N, groups.ngroups, nact
        if ng != len(geminal.gammas):
            raise ValueError(f"{ng} basis-function groups but {len(geminal.gammas)} gammas")
        self.groups = range(ng)
        self.pairs_T = geminal.pairs
        naux = fit.nbf
        self.naux = naux
        big = [f"{x}{G}" for x in "dZ" for G in self.groups] + ["coulomb"]
        self.store = ThreeIndexStore(len(big) * na * naux * N * DOUBLE, mode,
                                     "f12xg_3index")
        bo, bri, C = ri.obs_basis, ri.basis, ri.C
        Co = ri.C[ri.obs_idx][:, act]                        # orbital-basis AOs
        Cx = np.hstack([Co * groups.mask(G)[:, None] for G in self.groups])
        Cx_ri = np.zeros((bri.nbf, ng * na))
        Cx_ri[ri.obs_idx] = Cx
        S = np.asarray(_core.overlap(bri))
        Tx = C.T @ (S @ Cx_ri)                               # masked orbitals in the RI MOs
        Hx = ri.hj @ Tx                                      # hJ k^G in the RI MOs
        Hx_ri = C @ Hx
        sl = lambda G: slice(G * na, (G + 1) * na)  # noqa: E731
        coul = _core.TwoBodyOperator("coulomb")
        J = cho_factor(np.asarray(_core.two_body_2c(coul, fit)))

        def three(op, basis, C1, C2):  # (P|op|k q) -> (n1, naux, n2)
            R = np.asarray(_core.two_body_3c_mo(op, fit, basis, C1, C2))
            return R.reshape(naux, C1.shape[1], C2.shape[1]).transpose(1, 0, 2)

        def M(op):
            return np.asarray(_core.two_body_2c(op, fit))

        def robust(M, R, d):  # G~ = (P|op|rho) - 1/2 (P|op|Q) d_rho, M = (P|op|Q)
            return R - 0.5 * np.einsum("PQ,kQq->kPq", M, d, optimize=True)
        # large arrays: masked Coulomb fit, Coulomb, Z_G = sum_H G~ of f_GH for the mask H
        B = three(coul, bri, Cx_ri, C)                       # (ng nact, naux, N')
        d = np.stack([cho_solve(J, B[k]) for k in range(ng * na)])
        for G in self.groups:
            self.store.put(f"d{G}", d[sl(G)])
        self.store.put("coulomb", sum(B[sl(G)] for G in self.groups))
        del B
        Z = np.zeros_like(d)
        for T in geminal.pairs:
            op = geminal.operators(*T)["f12"]
            masks = sorted(set(T))
            R = three(op, bri, np.hstack([Cx_ri[:, sl(G)] for G in masks]), C)
            Gt = robust(M(op), R, np.vstack([d[sl(G)] for G in masks]))
            del R
            for n, X in enumerate(masks):  # to Z of the partner group of the mask X in T
                Z[sl(T[0] if X == T[1] else T[1])] += Gt[n * na:(n + 1) * na]
            del Gt
        for G in self.groups:
            self.store.put(f"Z{G}", Z[sl(G)])
        del Z
        # small arrays: fit coefficients of the masked active densities (k^G m), (k^G m^G1),
        # (k^G hJ m^G1), MO integrals Y[G, k, G1, m, H, l, H1, n] = (k^G m^G1|op|l^H n^H1)
        dm = d[:, :, act]
        dmm = np.einsum("kPq,qm->kPm", d, Tx, optimize=True)
        dmh = np.einsum("kPq,qm->kPm", d, Hx, optimize=True)
        del d

        def ein(a, ga, b, gb):  # (k m|l n) = sum_P a(k m) G~b(l n) + G~a(k m) b(l n)
            return np.einsum("kPm,lPn->kmln", a, gb, optimize=True) + \
                np.einsum("kPm,lPn->kmln", ga, b, optimize=True)

        def only(X, groups):  # masked orbitals of the needed groups (others skipped)
            return np.hstack([X[:, sl(G)] * (G in groups) for G in self.groups])

        def fg(T):
            op = geminal.operators(*T)["f12/r12"]
            Gt = robust(M(op), three(op, bo, only(Cx, T), Co), dm)
            return ein(dm, Gt, dm, Gt).reshape(ng, na, 1, na, ng, na, 1, na)

        CH = np.hstack([Cx_ri, Hx_ri])

        def prod(Tb, T):
            ops = geminal.product_operators(Tb, T)
            need = set(T) | set(Tb)  # groups of k^G, l^H
            op = ops["grad f12 . grad f12"]
            Gt = robust(M(op), three(op, bo, only(Cx, need), Cx), dmm)
            YU = ein(dmm, Gt, dmm, Gt).reshape((ng, na) * 4)
            # F2 and the hJ-transformed bra densities in one call (RI basis)
            op = ops["f12 x f12"]
            R, Mf = three(op, bri, only(Cx_ri, need), CH), M(op)
            Gt = robust(Mf, R[:, :, :ng * na], dmm)
            Gh = robust(Mf, R[:, :, ng * na:], dmh)
            del R
            YF = ein(dmm, Gt, dmm, Gt).reshape((ng, na) * 4)
            YH = ein(dmh, Gh, dmm, Gt).reshape((ng, na) * 4)
            return YU, YF, YH

        self.FG, self.U, self.F2, self.H1 = _small_arrays(geminal, na, fg, prod)
        self.time = time.time() - t0

    def pair(self, i, j):
        """Pair integrals of the active pair (i, j), i >= j (as ExactF12XGIntegrals.pair):
        F = sum_G d_Gi^T Z_Gj + Z_Gi^T d_Gj."""
        get = self.store.get
        F = np.zeros((self.N, self.N))
        for G in self.groups:
            F += get(f"d{G}", i).T @ get(f"Z{G}", j) + get(f"Z{G}", i).T @ get(f"d{G}", j)
        di = sum(get(f"d{G}", i) for G in self.groups)
        out = {"F": F, "G": di.T @ get("coulomb", j)}
        out.update(_blocks(self, i, j))
        return out

    def close(self):
        self.store.close()


def kinetic_weight(geminal, bra, ket):
    """Weight of the [F^2 bar] term with hJ on the bra orbitals (geminal f_a of the group
    pair bra) and the ket geminal f_b (group pair ket): 2 g_b / (g_a + g_b).

    Approximation C writes f_a T f_b as 1/2 [f_a, [T, f_b]] (U) plus F^2 terms with T on
    the orbitals. For f_a != f_b the symmetric form 1/2 (f_a f_b T + T f_a f_b) misses
    1/2 (f_a T f_b - f_b T f_a); for Slater functions f = -exp(-g r12)/g it is
    f_a grad f_b - f_b grad f_a = (1/g_b - 1/g_a) grad (f_a f_b g_a g_b / (g_a + g_b)), so
    f_a T f_b = U + (g_b f_a f_b T + g_a T f_a f_b) / (g_a + g_b) (T acting to the right,
    resp. on the bra); the local h - T and J cancel in the antisymmetric part, so hJ
    replaces T. B = 1/2 (A + A^T) then needs the weight 2 g_b / (g_a + g_b) on the term
    with hJ on the bra (1 for equal geminals). Fitted geminals use their nominal gamma.
    """
    ga, gb = geminal.geminal(*bra).gamma, geminal.geminal(*ket).gamma
    return 2.0 * gb / (ga + gb)


def _small_arrays(geminal, na, fg, prod):
    """FG, U, F2, H1 [k, l, m, n] (see ExactF12XGIntegrals) from MO integrals over the masked
    active orbitals, Y[G, k, G1, m, H, l, H1, n] = (k^G m^G1|op|l^H n^H1): fg(T) for f_T/r12
    (m, n unmasked: G1 = H1 = 0), prod(T', T) -> (Y of (grad f_T').(grad f_T), of f_T' f_T,
    of f_T' f_T with hJ m^G1) for each unordered pair of group pairs (the product operators
    are symmetric in T', T)."""
    def gsum(Y, ket, bra):
        """sum of Y[G, k, G1, m, H, l, H1, n] over the orientations (G, H) of the ket group
        pair and (G1, H1) of the bra group pair, as [k, l, m, n]"""
        return sum(Y[G, :, G1, :, H, :, H1, :] for G, H in {ket, ket[::-1]}
                   for G1, H1 in {bra, bra[::-1]}).transpose(0, 2, 1, 3)

    FG, U, F2, H1 = (np.zeros((na,) * 4) for _ in range(4))
    for T in geminal.pairs:
        FG += gsum(fg(T), T, (0, 0))
    for n, Tb in enumerate(geminal.pairs):
        for T in geminal.pairs[n:]:
            YU, YF, YH = prod(Tb, T)
            for Tk, Tbra in {(T, Tb), (Tb, T)}:
                U += gsum(YU, Tk, Tbra)
                F2 += gsum(YF, Tk, Tbra)
                H1 += kinetic_weight(geminal, Tbra, Tk) * gsum(YH, Tk, Tbra)
    return FG, U, F2, H1


def _blocks(ints, i, j):
    """Blocks over kl, mn in (ij, ji) of the arrays FG, U, F2, H1 (see
    ExactF12XGIntegrals.pair)."""
    basis = [(i, j)] if i == j else [(i, j), (j, i)]
    return {
        "f12/r12": np.array([[ints.FG[k, l, m, n] for m, n in basis] for k, l in basis]),
        "(grad f12)^2": np.array([[ints.U[m, n, k, l] for m, n in basis] for k, l in basis]),
        "F2": np.array([[ints.F2[m, n, k, l] for m, n in basis] for k, l in basis]),
        "F2bar": np.array([[ints.H1[m, n, k, l] + ints.H1[n, m, l, k] for m, n in basis]
                           for k, l in basis])}


# --- task mp2-f12xg-pilot ------------------------------------------------------------------

def task_mp2_f12xg_pilot(system, options, ctx):
    """F12XG pilot (and df- variant): MP2-F12 with fixed amplitudes and one geminal per
    pair of basis-function groups (RHF); as task mp2-f12 otherwise.

    Args:
        system: System (with an RHF wavefunction).
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    from .correlation import orbital_spaces, print_spaces
    from .geminal import XGGeminal
    from .mp2_f12 import (_approximation, _families, cabs_singles_energy, fitting_basis,
                          pair_energies, print_ri_space, report, ri_space)
    from .options import InputError
    out = ctx.out
    wf = system.wavefunction
    df = bool(options.get("df"))
    method = ("DF-" if df else "") + "MP2-F12XG"
    if wf is None:
        raise InputError(f"mp2-f12xg-pilot: needs an SCF task for system '{system.name}' "
                         "before")
    out.section(method + " (pilot)")
    spaces = orbital_spaces(wf, options["frozen"], options["deleted"], options["semicanonical"],
                            jk=system.scf_jk)
    if spaces.reference != "RHF":
        raise InputError("mp2-f12xg-pilot: closed-shell (RHF) references only")
    try:
        gammas = [float(g) for g in options["gammas"]]
        if len(gammas) != 2:
            raise ValueError(f"gammas: two values needed, got {len(gammas)}")
        groups = assign_bf_groups(system.molecule, system.basis, options["bf_groups"],
                                  options["core_valence_threshold"])
        geminal = XGGeminal(gammas, options["geminal"])
    except (ValueError, TypeError) as e:
        raise InputError(f"mp2-f12xg-pilot: {e}") from None
    approx = _approximation(options)
    if _families(approx)[1]:
        raise InputError("mp2-f12xg-pilot: approximation A not implemented")
    print_spaces(spaces, list(wf.irrep_names), out)
    for line in groups.summary(system.molecule, system.basis):
        out("  " + line)
    for line in geminal.describe():
        out("  geminal " + line)
    out.value("approximation", ", ".join(approx) + " (fixed amplitudes 1/2, 1/4; ansatz 3)")
    ri, cabs_name, how = ri_space(system, spaces, options, df, out)
    print_ri_space(ri, how, out)
    out.flush()

    a = spaces.alpha
    e_singles = cabs_singles_energy(ri, a, options, out)
    fit = None
    if df:
        fit = fitting_basis(system, options, out)
        ints = DFF12XGIntegrals(ri, a.nfrozen, a.nocc, fit, geminal, groups,
                                options["integrals"])
        out.value("3-index integrals", f"{ints.store.describe()}, {ints.time:.2f} s")
    else:
        ints = ExactF12XGIntegrals(ri, a.nfrozen, a.nocc, geminal, groups,
                                   options["threshold"], options["integrals"])
        out.value("pair integrals", f"exact (F12XG), {ints.store.describe()}, "
                  f"{ints.time:.2f} s")
    out.flush()
    res = pair_energies(ints, ri, a, options, out)
    sizes = [int(np.sum(groups.mask(G))) for G in range(groups.ngroups)]
    data = report(system, options, ctx, method, spaces, res, e_singles, "mp2_f12xg",
                  {"cabs_auxbasis": cabs_name, "frozen": a.nfrozen, "gammas": np.array(gammas),
                   "bf_groups": groups.labels + 1})
    return {"method": method, "reference": spaces.reference, **data, "gammas": gammas,
            "bf_groups": groups.how, "group_sizes": sizes, "cabs_auxbasis": cabs_name,
            "ncabs": ri.ncabs,
            **({"density_fitting": {"c_auxbasis": fit.name, "naux": fit.nbf}} if df else {})}
