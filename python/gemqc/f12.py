"""F12 integrals and pair intermediates with fixed amplitudes (shared F12 layer).

Orbitals of the RI space (cabs.RISpace): occupied o (frozen core first, then the active
occupied i, j, k, l), virtual a (orbital basis), CABS x; p' runs over all of them.

Density fitting (fitting basis of c_auxbasis, Coulomb metric J): fit coefficients
d_k = J^-1 (P|k p') and the Coulomb (P|k p'); robust fitting of the geminal operators
(Manby), stored as G~_k = (P|g|k p') - 1/2 (P|g|Q) d_k::

    (k p'|g|l q') = d_k^T (P|g|l q') + (P|g|k p')^T d_l - d_k^T (P|g|Q) d_l
                  = d_k^T G~_l + G~_k^T d_l

The 3-index arrays (active k, all p') live in a transform.ThreeIndexStore (memory/disk);
pair matrices <ij|g|p'q'> (N' x N') are assembled per pair.

Pair intermediates for kl, mn in {ij, ji} (FIX amplitudes need no others); P-hat = 1 - Q12
are the pairs (rs) of the orbital basis and (o x), (x o)::

    V^kl_ij = <kl|f12/r12|ij> - sum_{P-hat} <kl|F|pq><pq|r12^-1|ij>
    X^kl_mn = <kl|F^2|mn> - sum_{P-hat} <kl|F|pq><pq|F|mn>
    B = 1/2 S(A - Z - F_ab C^ab),  S M^kl_mn = M^kl_mn + M^mn_kl
    A = U + [F^2 bar] - sum_{P-hat} F-bar^{pq}_kl F^{pq}_mn - Y~

B is approximation C (Molpro 3C/3*C with NOZ, HY1; docs/theory.md): U = <kl|(grad_1
F)^2|mn>, bars a transformation with the Hartree operator h + J, Y~ the exchange terms
over the Q pairs (x a), (a x), (x y), Z the occupied-CABS Fock terms (omitted with NOZ /
GBC), C^ab_mn = f_ax F^xb_mn + F^ax_mn f_bx.
"""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import _core
from .transform import DOUBLE, ThreeIndexStore


class F12Integrals:
    """Density-fitted F12 integrals over the RI space for the active occupied orbitals.

    Args:
        ri: cabs.RISpace (orbitals: frozen, active occupied, virtual, CABS).
        nfrozen: Frozen core orbitals (the first ones).
        nact: Active occupied orbitals.
        fit: Fitting basis (BasisSet).
        geminal: geminal.Geminal.
        mode: Storage 'auto', 'memory' or 'disk'.
    """

    BIG = ("d", "coulomb", "f12", "f12^2")     # (nact, naux, N') arrays
    SMALL = ("f12/r12", "(grad f12)^2")        # (nact, naux, nact) arrays

    def __init__(self, ri, nfrozen, nact, fit, geminal, mode="auto"):
        t0 = time.time()
        self.ri, self.nfrozen, self.nact = ri, nfrozen, nact
        self.act = np.arange(nfrozen, nfrozen + nact)
        ops = geminal.operators()
        N = ri.C.shape[1]
        naux = fit.nbf
        self.naux, self.N = naux, N
        size = (len(self.BIG) * N + len(self.SMALL) * nact) * nact * naux * DOUBLE
        self.store = ThreeIndexStore(size, mode, "f12_integrals")
        Ca = ri.C[:, self.act]
        J = cho_factor(np.asarray(_core.two_body_2c(ops["coulomb"], fit)))

        def three(op, C2):  # (P|op|k q) -> (nact, naux, n2)
            R = np.asarray(_core.two_body_3c_mo(op, fit, ri.basis, Ca, C2))
            return R.reshape(naux, nact, C2.shape[1]).transpose(1, 0, 2)
        B = three(ops["coulomb"], ri.C)
        d = np.stack([cho_solve(J, B[k]) for k in range(nact)])
        self.store.put("d", d)
        self.store.put("coulomb", B)
        del B
        for name in ("f12", "f12^2"):
            M = np.asarray(_core.two_body_2c(ops[name], fit))
            G = three(ops[name], ri.C)
            self.store.put(name, G - 0.5 * np.einsum("PQ,kQq->kPq", M, d, optimize=True))
        da = d[:, :, self.act]
        for name in self.SMALL:
            M = np.asarray(_core.two_body_2c(ops[name], fit))
            G = three(ops[name], Ca)
            self.store.put(name, G - 0.5 * np.einsum("PQ,kQq->kPq", M, da, optimize=True))
        self.time = time.time() - t0

    def slices(self, k):
        """All 3-index slices of the active orbital k (0-based in the active space)."""
        return {name: self.store.get(name, k) for name in self.BIG + self.SMALL}

    def pair(self, i, j):
        """Pair integrals of the active pair (i, j), i >= j: F = <ij|F|p'q'>,
        G = <ij|r12^-1|p'q'> (N' x N'), F2c {'ij': <ij|F^2|p' l>, 'ji': <ji|F^2|p' l>}
        (l active), and the blocks <kl|g|mn> (kl, mn in (ij, ji)) of f12/r12 and
        (grad f12)^2."""
        si = self.slices(i)
        sj = si if i == j else self.slices(j)
        act = self.act

        def robust(name, a, b, cols=None):
            Ga, Gb, da, db = a[name], b[name], a["d"], b["d"]
            if cols is not None:
                Gb, db = Gb[:, cols], db[:, cols]
            return da.T @ Gb + Ga.T @ db

        def small(name, k, l, m, n):  # <kl|g|mn> from the slices of k and l
            sk, sl = (si if k == i else sj), (si if l == i else sj)
            return sk["d"][:, act[m]] @ sl[name][:, n] + sk[name][:, m] @ sl["d"][:, act[n]]
        basis = [(i, j)] if i == j else [(i, j), (j, i)]
        blk = {name: np.array([[small(name, k, l, m, n) for (m, n) in basis]
                               for (k, l) in basis]) for name in self.SMALL}
        return {"F": robust("f12", si, sj), "G": si["d"].T @ sj["coulomb"],
                "F2c": {"ij": robust("f12^2", si, sj, act), "ji": robust("f12^2", sj, si, act)},
                **blk}

    def close(self):
        self.store.close()


class ExactF12Integrals:
    """F12 pair integrals <ij|g|p'q'> with exact integrals, for the active pairs.

    The occupied orbitals have no CABS components, so the AO indices contracted with
    them run over the orbital basis only (C++ `pair_integrals`: K^ij(nu, si) =
    sum_{mu, la} (mu nu|g|la si) C_mu,i C_la,j), and each operator gets only the blocks it
    needs:

    - F (f12): nu, si in the RI basis -> <ij|F|p'q'> (N' x N');
    - G (Coulomb), F^2: nu in the RI, si in the orbital basis, ordered pairs (i, j) and
      (j, i) -> <ij|g|p' s> (s orbital basis), which gives the blocks (p' s) and, from the
      swapped pair, (o x) of G (the (v x) and (x y) blocks are not needed and zero) and
      the F^2 columns <ij|F^2|p' l>, <ji|F^2|p' l>;
    - f12/r12, (grad f12)^2: all indices in the orbital basis -> <ij|g|mn> (m, n active),
      with FockBuilder.pair_jk (permutational symmetry).

    Stored per pair i >= j (transform.ThreeIndexStore, memory/disk).

    Args:
        ri: cabs.RISpace.
        nfrozen: Frozen core orbitals.
        nact: Active occupied orbitals.
        geminal: geminal.Geminal.
        threshold: Integral screening.
        mode: Storage 'auto', 'memory' or 'disk'.
    """

    SMALL = ("f12/r12", "(grad f12)^2")

    def __init__(self, ri, nfrozen, nact, geminal, threshold=1e-12, mode="auto"):
        t0 = time.time()
        self.act = act = np.arange(nfrozen, nfrozen + nact)
        self.nact, self.N = nact, ri.C.shape[1]
        N, nmo = self.N, ri.nmo
        pairs = [(i, j) for i in range(nact) for j in range(i + 1)]
        self.index = {p: n for n, p in enumerate(pairs)}
        npair = len(pairs)
        size = npair * (2 * N * N + 2 * N * nact + 2 * nact * nact) * DOUBLE
        self.store = ThreeIndexStore(size, mode, "f12_pair_integrals")
        shapes = {"F": (npair, N, N), "G": (npair, N, N), "f12^2": (npair, N, 2 * nact),
                  **{name: (npair, nact, nact) for name in self.SMALL}}
        for name, shape in shapes.items():
            self.store.create(name, shape)
        ops = geminal.operators()
        bo, bri = ri.obs_basis, ri.basis
        C = ri.C                                  # RI AOs x all orbitals
        Cobs = ri.C[ri.obs_idx, :nmo]             # orbital-basis AOs x orbital-basis MOs
        Co = Cobs[:, act]                         # active occupied over the orbital basis
        limit = _core.memory()
        free = (limit - (0 if self.store.on_disk else size)) if limit else 0
        o = len(ri.occupied)

        def batches(plist, per):
            nb = max(1, int(0.5 * free // per)) if limit and free > 0 else len(plist)
            return [plist[k:k + nb] for k in range(0, len(plist), nb)]
        # F: both free indices in the RI basis
        for chunk in batches(pairs, bri.nbf ** 2 * DOUBLE):
            K = _core.pair_integrals(ops["f12"], bo, bri, bri, Co, chunk, threshold)
            for (i, j), Kij in zip(chunk, K):
                self.store.set("F", self.index[(i, j)], C.T @ np.asarray(Kij) @ C)
        # G and F^2: <ij|g|p' s>, ordered pairs
        ordered = [(i, j) for i in range(nact) for j in range(nact)]
        for name in ("coulomb", "f12^2"):
            half = {}
            for chunk in batches(ordered, bri.nbf * bo.nbf * DOUBLE):
                K = _core.pair_integrals(ops[name], bo, bri, bo, Co, chunk, threshold)
                for (i, j), Kij in zip(chunk, K):
                    half[(i, j)] = C.T @ np.asarray(Kij) @ Cobs  # (N', nmo)
            for (i, j), n in self.index.items():
                if name == "coulomb":
                    G = np.zeros((N, N))
                    G[:, :nmo] = half[(i, j)]
                    G[:o, nmo:] = half[(j, i)][nmo:, :o].T  # <ij|o x> = <ji|x o>
                    self.store.set("G", n, G)
                else:
                    self.store.set("f12^2", n, np.hstack([half[(i, j)][:, act],
                                                          half[(j, i)][:, act]]))
            del half
        # small: all in the orbital basis (pair_jk uses the permutational symmetry)
        Ca = Cobs[:, act]
        for name in self.SMALL:
            _, K = _core.FockBuilder(bo, threshold, ops[name]).pair_jk(Co, pairs)
            for (i, j), Kij in zip(pairs, K):
                self.store.set(name, self.index[(i, j)], Ca.T @ np.asarray(Kij) @ Ca)
        self.time = time.time() - t0

    def pair(self, i, j):
        """Pair integrals of the active pair (i, j), i >= j (as F12Integrals.pair)."""
        n, na = self.index[(i, j)], self.nact
        F2 = self.store.get("f12^2", n)
        out = {"F": self.store.get("F", n), "G": self.store.get("G", n),
               "F2c": {"ij": F2[:, :na], "ji": F2[:, na:]}}
        for name in self.SMALL:
            S = self.store.get(name, n)  # <ij|g|mn>
            out[name] = (np.array([[S[i, j]]]) if i == j else
                         np.array([[S[i, j], S[j, i]], [S[j, i], S[i, j]]]))
        return out

    def close(self):
        self.store.close()


class PairIntermediates:
    """V, X, B and coupling quantities of the pairs, for the FIX energy.

    Args:
        ints: F12Integrals.
        ri: cabs.RISpace.
        noz: Omit the Z terms (GBC in B).
        hy1: Omit the exchange terms Y~ (Molpro HY1).
        nvir_act: Active virtual orbitals (the first virtuals; the rest are deleted).
    """

    def __init__(self, ints, ri, noz=False, hy1=False, nvir_act=None):
        self.ints, self.ri, self.noz, self.hy1 = ints, ri, noz, hy1
        nocc, nmo, N = len(ri.occupied), ri.nmo, ri.C.shape[1]
        self.o, self.v, self.x = slice(0, nocc), slice(nocc, nmo), slice(nmo, N)
        self.nvir_act = nmo - nocc if nvir_act is None else nvir_act
        M = np.zeros((N, N), bool)  # P-hat pairs
        M[:nmo, :nmo] = True
        M[:nocc, nmo:] = True
        M[nmo:, :nocc] = True
        self.P = M
        self.f, self.hj, self.k = ri.fock, ri.hj, ri.exchange

    def pair(self, i, j):
        """Intermediates of the active pair (i, j), i >= j (0-based active indices).

        Returns:
            dict: basis 'kl' (list of (k, l)), V (vector over kl), X, B3C, B3sC (matrices
            over kl, mn: with and without the F C term), C (coupling matrix C^ij_ab over
            the active virtuals), K = (ia|jb) over the active virtuals.
        """
        ints = self.ints
        pi = ints.pair(i, j)
        F, G = pi["F"], pi["G"]
        basis = [(i, j)] if i == j else [(i, j), (j, i)]
        Fk = [F] if i == j else [F, F.T]
        P, o, v, x, f = self.P, self.o, self.v, self.x, self.f
        act = ints.act
        n = len(basis)
        FG, U = pi["f12/r12"], pi["(grad f12)^2"]
        # <kl|F^2|mn> and the [F^2 bar] term of A: given as blocks (F12XG) or from the
        # F^2 columns <kl|F^2|p' n>
        if "F2" in pi:
            F2, F2bar = pi["F2"], pi["F2bar"]
        else:
            F2cols = {(i, j): pi["F2c"]["ij"], (j, i): pi["F2c"]["ji"]}
            F2 = np.array([[F2cols[kl][act[m], n_] for (m, n_) in basis] for kl in basis])
            hj = self.hj
            F2bar = np.array([[hj[act[k]] @ F2cols[(m_, n_)][:, l]
                               + hj[act[l]] @ F2cols[(n_, m_)][:, k]
                               for (m_, n_) in basis] for (k, l) in basis])
        PF = [Fm * P for Fm in Fk]
        V = np.array([FG[a, 0] - np.sum(PF[a] * G) for a in range(n)])
        X = F2 - np.array([[np.sum(PF[a] * Fk[b]) for b in range(n)] for a in range(n)])
        # A = U + [F^2 bar] - <F-bar, F>_P-hat - Y~
        hj, kx = self.hj, self.k
        A = U + F2bar
        Fbar = [hj @ Fm + Fm @ hj for Fm in Fk]
        for a in range(n):
            for b in range(n):
                A[a, b] -= np.sum(Fbar[a] * PF[b])
        if not self.hy1:
            Q = ~P
            for a in range(n):
                Ft = kx @ Fk[a] + Fk[a] @ kx
                for b in range(n):
                    A[a, b] -= np.sum(Ft[Q] * Fk[b][Q])
        if not self.noz:
            fxo = f[x, o]
            for a in range(n):
                Fa_ = Fk[a]
                chk = (fxo.T @ Fa_[x, x], Fa_[x, x] @ fxo, fxo.T @ Fa_[x, v], Fa_[v, x] @ fxo)
                for b in range(n):
                    Fb_ = Fk[b]
                    A[a, b] -= (np.sum(chk[0] * Fb_[o, x]) + np.sum(chk[1] * Fb_[x, o])
                                + np.sum(chk[2] * Fb_[o, v]) + np.sum(chk[3] * Fb_[v, o]))
        # coupling: C^mn_ab = f_ax F^mn_xb + F^mn_ax f_bx (all virtuals for B)
        fvx = f[v, x]
        Cm = [fvx @ Fm[x, v] + Fm[v, x] @ fvx.T for Fm in Fk]
        FC = np.array([[np.sum(Fk[a][v, v] * Cm[b]) for b in range(n)] for a in range(n)])
        B3sC = 0.5 * (A + A.T)
        B3C = B3sC - 0.5 * (FC + FC.T)
        nv = self.nvir_act
        return {"basis": basis, "V": V, "X": X, "B3sC": B3sC, "B3C": B3C,
                "C": Cm[0][:nv, :nv], "K": G[v, v][:nv, :nv]}
