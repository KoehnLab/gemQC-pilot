"""Generic two-electron operators (Coulomb, Slater geminal, Yukawa) against independent
references: closed-form s-function integrals, quadrature of high angular momentum shells
in the potential of an s Gaussian, and relations between the operators."""

import numpy as np
import pytest
from scipy.integrate import lebedev_rule, quad
from scipy.special import erfc, erfcx

from gemqc import _core

OPS = [("coulomb", 0.0), ("stg", 1.0), ("stg", 2.0), ("stg_x_coulomb", 1.0),
       ("stg_x_coulomb", 0.3)]


# --- reference: potential of a Gaussian exp(-q |r2 - Q|^2) for g(r12) ---------------------

def _radial(kind, zeta, q, s):
    """int_0^inf r g(r) exp(-q (r - s)^2) dr (closed form, numerically stable)."""
    s = np.asarray(s, dtype=float)
    m = s - 0.5 * zeta / q
    sq = np.sqrt(q)
    # e^{-q s^2} e^{q m^2} erfc(-m sqrt q), split by the sign of m
    pos = m >= 0
    br = np.empty_like(m)
    br[pos] = np.exp(q * (m[pos] ** 2 - s[pos] ** 2)) * (2.0 - erfc(m[pos] * sq))
    br[~pos] = np.exp(-q * s[~pos] ** 2) * erfcx(-m[~pos] * sq)
    if kind == "stg_x_coulomb" or kind == "coulomb":  # r g(r) = exp(-zeta r)
        return 0.5 * np.sqrt(np.pi / q) * br
    return np.exp(-q * s * s) / (2.0 * q) + m * 0.5 * np.sqrt(np.pi / q) * br  # r e^{-zeta r}


def _potential_small(kind, zeta, q, R):
    """The potential for small R by quadrature of r g(r) e^{-q(r^2+R^2)} 2 sinh(2qrR)/R."""
    g = _g(kind, zeta)

    def f(r):
        x = 2 * q * r * R
        sh = 2 * np.sinh(x) / R if R > 0 else 4 * q * r
        return r * g(r) * np.exp(-q * (r * r + R * R)) * sh
    return np.pi / q * quad(f, 0, 40 / np.sqrt(q), epsabs=0, epsrel=1e-13, limit=400)[0]


def potential(kind, zeta, q, R):
    """int exp(-q |r - R|^2) g(|r|) d^3r for distances R (array); closed form, quadrature
    for R < 0.05 (cancellation)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    out = np.empty_like(R)
    big = R >= 0.05
    Rb = R[big]
    out[big] = np.pi / (q * Rb) * (_radial(kind, zeta, q, Rb) - _radial(kind, zeta, q, -Rb))
    out[~big] = [_potential_small(kind, zeta, q, x) for x in R[~big]]
    return out if out.size > 1 else out[0]


def _g(kind, zeta):
    if kind == "coulomb":
        return lambda r: 1.0 / r
    if kind == "stg":
        return lambda r: np.exp(-zeta * r)
    return lambda r: np.exp(-zeta * r) / r


@pytest.mark.parametrize("kind,zeta", OPS)
def test_potential_closed_form(kind, zeta):
    """The closed-form potential against direct quadrature."""
    g, q = _g(kind, zeta), 0.7
    for R in (0.3, 1.5, 4.0):
        f = lambda r: r * g(r) * (np.exp(-q * (r - R) ** 2) - np.exp(-q * (r + R) ** 2))
        ref = np.pi / (q * R) * quad(f, 0, R + 30, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
        assert potential(kind, zeta, q, R) == pytest.approx(ref, rel=1e-11)


# --- basis helpers ------------------------------------------------------------------------

def _basis(centres, shells):
    """Molecule with atoms (Z, xyz) and per-Z shell definitions [(l, exponent)]."""
    mol = _core.Molecule()
    for z, r in centres:
        mol.add_atom(z, r)
    lib = {z: [_core.ShellDef(l, True, [a], [1.0]) for l, a in sh] for z, sh in shells.items()}
    return mol, _core.BasisSet(mol, lib)


def _norm(a):
    return (2.0 * a / np.pi) ** 0.75


def _ssss(kind, zeta, prims):
    """(ab|g|cd) of normalized s primitives [(exponent, centre)] (closed form)."""
    (a, A), (b, B), (c, C), (d, D) = [(e, np.asarray(x)) for e, x in prims]
    p, q = a + b, c + d
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    K = np.exp(-a * b / p * np.sum((A - B) ** 2) - c * d / q * np.sum((C - D) ** 2))
    alpha = p * q / (p + q)
    pre = _norm(a) * _norm(b) * _norm(c) * _norm(d) * K * (np.pi ** 2 / (p * q)) ** 1.5
    return pre * (alpha / np.pi) ** 1.5 * potential(kind, zeta, alpha, np.linalg.norm(P - Q))


@pytest.mark.parametrize("kind,zeta", OPS)
def test_s_functions(kind, zeta):
    """4-, 3- and 2-centre s integrals against the closed form."""
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.9, 0.3], [1.1, -0.4, 0.8], [-0.5, 0.6, 2.0]]
    ex = [1.3, 0.45, 0.8, 2.1]
    mol, bs = _basis([(z + 1, r) for z, r in enumerate(xyz)],
                     {z + 1: [(0, a)] for z, a in enumerate(ex)})
    op = _core.TwoBodyOperator(kind, zeta, 1.0)
    V = _core.two_body_4c(op, bs)
    prims = list(zip(ex, xyz))
    for idx in [(0, 1, 2, 3), (0, 0, 3, 3), (1, 2, 1, 2), (3, 2, 0, 0), (2, 2, 2, 2)]:
        ref = _ssss(kind, zeta, [prims[i] for i in idx])
        assert V[idx] == pytest.approx(ref, rel=1e-12, abs=1e-14)
    # 3- and 2-centre: a unit function has norm 1 and exponent 0 (P = fitting function)
    V3 = _core.two_body_3c(op, bs, bs).reshape(4, 4, 4)
    V2 = _core.two_body_2c(op, bs)
    for P, (i, j) in [(0, (1, 2)), (3, (3, 0)), (2, (2, 2))]:
        a, A = prims[P]
        p = a
        b, B = prims[i]
        c, C = prims[j]
        q = b + c
        Q = (b * np.asarray(B) + c * np.asarray(C)) / q
        K = np.exp(-b * c / q * np.sum((np.asarray(B) - np.asarray(C)) ** 2))
        alpha = p * q / (p + q)
        ref = (_norm(a) * _norm(b) * _norm(c) * K * (np.pi ** 2 / (p * q)) ** 1.5
               * (alpha / np.pi) ** 1.5 * potential(kind, zeta, alpha,
                                                    np.linalg.norm(np.asarray(A) - Q)))
        assert V3[P, i, j] == pytest.approx(ref, rel=1e-12)
    for P, Q in [(0, 1), (2, 3), (1, 1)]:
        (a, A), (b, B) = prims[P], prims[Q]
        alpha = a * b / (a + b)
        ref = (_norm(a) * _norm(b) * (np.pi ** 2 / (a * b)) ** 1.5 * (alpha / np.pi) ** 1.5
               * potential(kind, zeta, alpha, np.linalg.norm(np.asarray(A) - np.asarray(B))))
        assert V2[P, Q] == pytest.approx(ref, rel=1e-12)


# --- high angular momentum: quadrature in the potential of an s Gaussian -----------------

def _grid(centre, nrad=160, order=131, rm=1.0):
    """Atom-centred product grid (Becke radial map, Lebedev angular): points, weights."""
    x, wx = np.polynomial.legendre.leggauss(nrad)
    r = rm * (1 + x) / (1 - x)
    wr = wx * 2 * rm / (1 - x) ** 2 * r ** 2
    u, wu = lebedev_rule(order)
    pts = (r[:, None, None] * u.T[None, :, :]).reshape(-1, 3) + centre
    return pts, (wr[:, None] * wu[None, :]).ravel()


@pytest.mark.parametrize("kind,zeta", OPS)
def test_high_angular_momentum(kind, zeta):
    """(ab|g|ss) with a, b on one centre up to l = 7, (P|g|ss) and (P|g|s) (l = 7, the
    BasisSet limit; libint's 3-centre code goes to l = 8 for fitting functions):
    numerical quadrature of rho_ab (or P) times the potential of the s pair."""
    A, Q = np.zeros(3), np.array([0.4, -0.7, 1.1])
    q = 0.9  # s on Q
    mol, bs = _basis([(1, A), (2, Q)], {1: [(7, 1.1), (6, 0.8), (3, 0.5), (7, 1.3)],
                                        2: [(0, q)]})
    op = _core.TwoBodyOperator(kind, zeta, 1.0)
    pts, w = _grid(A)
    chi = _core.basis_values(bs, pts)
    off = bs.shell_offsets
    i7, i6, i3, i7b, s = (slice(off[k], off[k] + n) for k, n in
                         enumerate([15, 13, 7, 15, 1]))
    # potential of the s pair (s on Q, squared): N^2 int exp(-2q |r2 - Q|^2) g
    phi_pair = _norm(q) ** 2 * potential(kind, zeta, 2 * q, np.linalg.norm(pts - Q, axis=1))
    phi_one = _norm(q) * potential(kind, zeta, q, np.linalg.norm(pts - Q, axis=1))
    ns = s.start
    # 4-centre classes (a b | s s)
    V = _core.two_body_4c(op, bs)
    for ia, ib in [(i7, i6), (i7, i7), (i6, i3), (i3, i3)]:
        ref = np.einsum("p,pa,pb->ab", w * phi_pair, chi[:, ia], chi[:, ib])
        assert np.abs(V[ia, ib, ns, ns] - ref).max() < 1e-11 * max(1.0, np.abs(ref).max())
    # 3-centre (P | s s) and (P | a s)? keep the pair spherical: (P|ss)
    V3 = _core.two_body_3c(op, bs, bs).reshape(bs.nbf, bs.nbf, bs.nbf)
    for iP in (i7b, i7, i3):
        ref = np.einsum("p,pa->a", w * phi_pair, chi[:, iP])
        assert np.abs(V3[iP, ns, ns] - ref).max() < 1e-11 * max(1.0, np.abs(ref).max())
    # 2-centre (P|Q) with Q the s function
    V2 = _core.two_body_2c(op, bs)
    for iP in (i7b, i6):
        ref = np.einsum("p,pa->a", w * phi_one, chi[:, iP])
        assert np.abs(V2[iP, ns] - ref).max() < 1e-11 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("kind,zeta", OPS)
def test_auxiliary_l8(kind, zeta):
    """Fitting functions up to l = 8 (2-/3-centre integrals of the custom libint):
    (P|g|ss) and (P|g|s) by quadrature; orbital shells are limited to l = 7."""
    A, Q = np.zeros(3), np.array([0.4, -0.7, 1.1])
    q = 0.9
    mol = _core.Molecule()
    mol.add_atom(1, A)
    mol.add_atom(2, Q)
    aux = _core.BasisSet(mol, {1: [_core.ShellDef(8, True, [1.1], [1.0]),
                                   _core.ShellDef(5, True, [0.7], [1.0])], 2: []}, "aux", True)
    bs = _core.BasisSet(mol, {1: [], 2: [_core.ShellDef(0, True, [q], [1.0])]})
    assert aux.max_l == 8
    with pytest.raises(ValueError, match="exceeds libint limit"):
        _core.BasisSet(mol, {1: [_core.ShellDef(8, True, [1.1], [1.0])], 2: []})
    op = _core.TwoBodyOperator(kind, zeta, 1.0)
    pts, w = _grid(A)
    chi = _core.basis_values(aux, pts)
    r = np.linalg.norm(pts - Q, axis=1)
    phi_pair = _norm(q) ** 2 * potential(kind, zeta, 2 * q, r)
    phi_one = _norm(q) * potential(kind, zeta, q, r)
    V3 = np.asarray(_core.two_body_3c(op, aux, bs)).reshape(aux.nbf, 1, 1)
    V2 = np.asarray(_core.two_body_2c(op, aux, bs))
    ref3 = np.einsum("p,pa->a", w * phi_pair, chi)
    ref2 = np.einsum("p,pa->a", w * phi_one, chi)
    assert np.abs(V3[:, 0, 0] - ref3).max() < 1e-11 * max(1.0, np.abs(ref3).max())
    assert np.abs(V2[:, 0] - ref2).max() < 1e-11 * max(1.0, np.abs(ref2).max())


# --- relations between operators ----------------------------------------------------------

def _mixed():
    mol, bs = _basis([(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5]), (3, [-0.8, 0.1, 0.9])],
                     {1: [(0, 1.2), (2, 0.7), (4, 1.5)], 2: [(1, 0.6), (3, 0.9)],
                      3: [(5, 1.1), (0, 0.3)]})
    return mol, bs


def test_zeta_derivative():
    """exp(-zeta r) = -d/dzeta exp(-zeta r)/r (central differences, all classes)."""
    _, bs = _mixed()
    zeta, h = 1.1, 1e-4
    stg = _core.two_body_4c(_core.TwoBodyOperator("stg", zeta), bs)
    yp = _core.two_body_4c(_core.TwoBodyOperator("stg_x_coulomb", zeta + h), bs)
    ym = _core.two_body_4c(_core.TwoBodyOperator("stg_x_coulomb", zeta - h), bs)
    assert np.abs(stg + (yp - ym) / (2 * h)).max() < 1e-8
    s3 = _core.two_body_3c(_core.TwoBodyOperator("stg", zeta), bs, bs)
    y3 = [_core.two_body_3c(_core.TwoBodyOperator("stg_x_coulomb", zeta + d), bs, bs)
          for d in (h, -h)]
    assert np.abs(s3 + (y3[0] - y3[1]) / (2 * h)).max() < 1e-8


def test_small_zeta_and_scale():
    """exp(-zeta r)/r -> 1/r - zeta (S x S) for small zeta; scale; symmetry; Coulomb = eri."""
    _, bs = _mixed()
    S = np.asarray(_core.overlap(bs))
    coul = _core.two_body_4c(_core.TwoBodyOperator("coulomb"), bs)
    assert np.abs(coul - _core.eri(bs)).max() < 1e-14
    zeta = 1e-3  # error O(zeta^2 <r12>)
    y = _core.two_body_4c(_core.TwoBodyOperator("stg_x_coulomb", zeta), bs)
    assert np.abs(y - (coul - zeta * np.einsum("pq,rs->pqrs", S, S))).max() < 1e-5
    with pytest.raises(ValueError):
        _core.two_body_2c(_core.TwoBodyOperator("stg", 1e-4), bs)
    st = _core.two_body_4c(_core.TwoBodyOperator("stg", 0.8, -1.0 / 0.8), bs)
    s1 = _core.two_body_4c(_core.TwoBodyOperator("stg", 0.8), bs)
    assert np.abs(st + s1 / 0.8).max() < 1e-15
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        assert np.abs(s1 - s1.transpose(perm)).max() < 1e-12  # interpolation


def test_mixed_bases():
    """Mixed basis sets give the sub-blocks of the union basis."""
    mol, union = _basis([(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5])],
                        {1: [(0, 1.2), (2, 0.7), (1, 0.4), (3, 0.9)],
                         2: [(1, 0.6), (0, 0.25)]})
    _, b1 = _basis([(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5])],
                   {1: [(0, 1.2), (2, 0.7)], 2: [(1, 0.6)]})
    _, b2 = _basis([(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5])],
                   {1: [(1, 0.4), (3, 0.9)], 2: [(0, 0.25)]})
    op = _core.TwoBodyOperator("stg", 1.0)
    U = _core.two_body_4c(op, union)
    # union order per atom: atom 1 (b1 shells, b2 shells), atom 2 (b1, b2)
    idx1 = np.r_[0:6, 6 + 3 + 7:6 + 3 + 7 + 3]
    idx2 = np.r_[6:16, 19:20]
    M = _core.two_body_4c(op, b1, b2, b1, b2)
    assert np.abs(M - U[np.ix_(idx1, idx2, idx1, idx2)]).max() < 1e-14
    M3 = _core.two_body_3c(op, b2, b1, b2).reshape(len(idx2), len(idx1), len(idx2))
    U3 = _core.two_body_3c(op, union, union).reshape(20, 20, 20)
    assert np.abs(M3 - U3[np.ix_(idx2, idx1, idx2)]).max() < 1e-14


@pytest.mark.parametrize("kind,zeta", [("stg", 1.0), ("coulomb", 0.0)])
def test_mo_integrals(kind, zeta):
    """mo_integrals against the transformed AO integrals: all indices equal (8-fold
    symmetry) with masked coefficients, symmetric ket only, and no symmetry (mixed bases)."""
    centres = [(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5])]
    _, bs = _basis(centres, {1: [(0, 1.2), (2, 0.7), (1, 0.4), (3, 0.9)],
                             2: [(1, 0.6), (0, 0.25)]})
    _, b2 = _basis(centres, {1: [(1, 0.4), (3, 0.9)], 2: [(0, 0.25)]})
    op = _core.TwoBodyOperator(kind, zeta)
    rng = np.random.default_rng(7)
    n = bs.nbf
    C = rng.standard_normal((n, 3))
    mask = np.zeros(n, bool)
    mask[[0, 1, 2, 3, 4, 5, 16, 17, 18]] = True  # s, d on atom 1 and p on atom 2
    Cx = np.hstack([C * mask[:, None], C * ~mask[:, None]])
    V = _core.two_body_4c(op, bs)
    ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", V, Cx, Cx, Cx, Cx, optimize=True)
    Y = _core.mo_integrals(op, bs, Cx, bs, Cx, bs, Cx, bs, Cx)
    assert np.abs(Y - ref).max() < 1e-12 * np.abs(ref).max()
    # only some blocks (G, G1, H, H1) of the masks (3 columns each) needed
    blocks = [(0, 1, 1, 0), (1, 1, 0, 0)]
    Y = _core.mo_integrals(op, bs, Cx, bs, Cx, bs, Cx, bs, Cx, 1e-12, blocks, 3, 2)
    for t in blocks:
        sl = np.ix_(*[range(3 * g, 3 * g + 3) for g in t])
        assert np.abs(Y[sl] - ref[sl]).max() < 1e-12 * np.abs(ref).max()
    D =rng.standard_normal((n, 2))
    ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", V, Cx, D, Cx, Cx, optimize=True)
    Y = _core.mo_integrals(op, bs, Cx, bs, D, bs, Cx, bs, Cx)
    assert np.abs(Y - ref).max() < 1e-12 * np.abs(ref).max()
    E = rng.standard_normal((b2.nbf, 4))
    V2 = _core.two_body_4c(op, bs, b2, bs, bs)
    ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", V2, Cx, E, C, Cx, optimize=True)
    Y = _core.mo_integrals(op, bs, Cx, b2, E, bs, C, bs, Cx)
    assert np.abs(Y - ref).max() < 1e-12 * np.abs(ref).max()


# --- contracted Gaussian geminals (STG-nG) -------------------------------------------------

def _gauss_potential(kind, gtg, alpha, R):
    """int exp(-alpha |r - R|^2) g(|r|) d^3r for g = sum c exp(-a r^2) (cgtg), g/r
    (cgtg_x_coulomb) or (grad g)^2 = sum 4 a_i a_j c_i c_j r^2 exp(-(a_i+a_j) r^2)."""
    from scipy.special import erf
    if kind == "delcgtg2":
        terms = [(a1 + a2, 4 * a1 * a2 * c1 * c2) for a1, c1 in gtg for a2, c2 in gtg]
    else:
        terms = gtg
    tot = 0.0
    for a, c in terms:
        s = alpha + a
        Rp = alpha * R / s  # centre of the product Gaussian
        pre = c * (np.pi / s) ** 1.5 * np.exp(-alpha * a / s * R * R)
        if kind == "cgtg":
            tot += pre
        elif kind == "cgtg_x_coulomb":
            tot += pre * (erf(np.sqrt(s) * Rp) / Rp if Rp > 1e-8 else 2 * np.sqrt(s / np.pi))
        else:
            tot += pre * (Rp * Rp + 1.5 / s)
    return tot


@pytest.mark.parametrize("kind", ["cgtg", "cgtg_x_coulomb", "delcgtg2"])
def test_gaussian_geminal_kernels(kind):
    """cgtg, cgtg_x_coulomb and delcgtg2 for s functions (4-, 3-, 2-centre) against the
    closed forms."""
    gtg = [(0.3, 0.4), (1.7, 0.25), (9.0, -0.1)]
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.9, 0.3], [1.1, -0.4, 0.8], [-0.5, 0.6, 2.0]]
    ex = [1.3, 0.45, 0.8, 2.1]
    _, bs = _basis([(z + 1, r) for z, r in enumerate(xyz)],
                   {z + 1: [(0, a)] for z, a in enumerate(ex)})
    op = _core.TwoBodyOperator(kind, 0.0, 0.7, gtg)
    V, V3, V2 = (_core.two_body_4c(op, bs), _core.two_body_3c(op, bs, bs).reshape(4, 4, 4),
                 _core.two_body_2c(op, bs))

    def pair(i, j):  # exponent, centre and prefactor of the product of s primitives
        (a, A), (b, B) = (ex[i], np.asarray(xyz[i])), (ex[j], np.asarray(xyz[j]))
        p = a + b
        return p, (a * A + b * B) / p, _norm(a) * _norm(b) * np.exp(-a * b / p * np.sum((A - B) ** 2))

    def single(i):
        return ex[i], np.asarray(xyz[i]), _norm(ex[i])

    def ref(x, y):
        (p, P, kp), (q, Q, kq) = x, y
        alpha = p * q / (p + q)
        return (0.7 * kp * kq * (np.pi ** 2 / (p * q)) ** 1.5 * (alpha / np.pi) ** 1.5
                * _gauss_potential(kind, gtg, alpha, np.linalg.norm(P - Q)))
    for i, j, k, l in [(0, 1, 2, 3), (0, 0, 3, 3), (1, 2, 1, 2), (2, 2, 2, 2)]:
        assert V[i, j, k, l] == pytest.approx(ref(pair(i, j), pair(k, l)), rel=1e-12)
    for P, i, j in [(0, 1, 2), (3, 3, 0), (2, 2, 2)]:
        assert V3[P, i, j] == pytest.approx(ref(single(P), pair(i, j)), rel=1e-12)
    for P, Q in [(0, 1), (2, 3), (1, 1)]:
        assert V2[P, Q] == pytest.approx(ref(single(P), single(Q)), rel=1e-12)


def test_stg_fit_molpro():
    """The STG-6G fit for gamma = 1 reproduces Molpro's printed exponents and coefficients;
    fits for other gamma and n are stationary and close to exp(-gamma r)."""
    from gemqc.geminal import stg_gtg_fit
    a, c = stg_gtg_fit(1.0, 6)
    assert np.allclose(np.round(a, 5), [0.19532, 0.81920, 2.85917, 9.50073, 35.69989,
                                        197.79328], atol=1e-12)
    assert np.allclose(np.round(c, 5), [0.27070, 0.30552, 0.18297, 0.10986, 0.06810,
                                        0.04224], atol=1e-12)
    r = np.linspace(0.3, 3.0, 50)
    for gamma, n in [(1.4, 6), (0.8, 4)]:
        a, c = stg_gtg_fit(gamma, n)
        fit = np.exp(-np.outer(r * r, a)) @ c
        assert np.abs(fit - np.exp(-gamma * r)).max() < (5e-3 if n == 6 else 2e-2)


def test_f12_operators():
    """F12 operator sets: exact Slater (closed form) and STG-nG (close to it, explicit
    exponents and coefficients)."""
    from gemqc.geminal import NAMES, Geminal, f12_operators
    gamma = 1.3
    ops = f12_operators(gamma, fit="exact")
    assert tuple(ops) == NAMES
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.9, 0.3]]
    ex = [1.3, 0.45]
    _, bs = _basis([(1, xyz[0]), (2, xyz[1])], {1: [(0, ex[0]), (2, 0.6)], 2: [(0, ex[1])]})
    prims = [(ex[i], xyz[i]) for i in (0, 1, 0, 1)]
    ref = {"coulomb": _ssss("coulomb", 0.0, prims),
           "f12": -_ssss("stg", gamma, prims) / gamma,
           "f12/r12": -_ssss("stg_x_coulomb", gamma, prims) / gamma,
           "f12^2": _ssss("stg", 2 * gamma, prims) / gamma ** 2,
           "(grad f12)^2": _ssss("stg", 2 * gamma, prims)}
    # s functions first in each atom: index 0 (atom 1 s), 6 (atom 2 s)
    for name, op in ops.items():
        assert _core.two_body_4c(op, bs)[0, 6, 0, 6] == pytest.approx(ref[name], rel=1e-12)
    # STG-6G approximates the Slater integrals
    fitted = f12_operators(gamma, fit=6)
    for name in NAMES:
        e, f = (_core.two_body_4c(o[name], bs) for o in (ops, fitted))
        assert np.abs(e - f).max() < 1e-2 * np.abs(e).max()  # 5e-4 ... 4e-3
    # explicit exponents and coefficients are used as given
    gem = Geminal.make(1.0, exponents=[0.5, 3.0], coefficients=[0.6, 0.3])
    assert not gem.exact and list(gem.operators()["f12"].gtg) == [(0.5, 0.6), (3.0, 0.3)]
    assert "STG-2G" in gem.describe()
    with pytest.raises(ValueError):
        f12_operators(0.0)
    with pytest.raises(ValueError):
        Geminal.make(1.0, exponents=[1.0])
