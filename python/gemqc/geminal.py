"""Slater-type geminal f12 = -exp(-gamma r12)/gamma and the two-electron operators of
F12 theory, as scaled libint operators (`_core.TwoBodyOperator`).

The Slater function is either used exactly (libint `stg` integrals) or, as in Molpro,
Turbomole and Dalton, represented by contracted Gaussian geminals (STG-nG),
exp(-gamma r) ~ sum_i c_i exp(-a_i r^2), with a_i and c_i from a least-squares fit with
the weight W(r) = r^2 exp(-(pi gamma^2)^(1/3) r^2) (this reproduces Molpro's STG-6G).

With f12' = exp(-gamma r12): (grad_1 f12)^2 = exp(-2 gamma r12), and the double
commutator [f12, [T1 + T2, f12]] = (grad_1 f12)^2 + (grad_2 f12)^2 (T: kinetic energy
operator). For the Gaussian expansion g, f12 = -g/gamma and (grad_1 f12)^2 =
(grad_1 g)^2/gamma^2 (libint `delcgtg2`), f12^2 = g^2/gamma^2 (expanded in pairs).

XGGeminal: the extended geminal of F12XG, one geminal per pair of basis-function groups.
"""

from dataclasses import dataclass

import numpy as np

from . import _core

#: operator names of f12_operators
NAMES = ("coulomb", "f12", "f12/r12", "f12^2", "(grad f12)^2")

#: Molpro's STG-6G fit for gamma = 1 (printed digits; starting point of the fits)
_REF6 = (np.array([0.19532, 0.81920, 2.85917, 9.50073, 35.69989, 197.79328]),
         np.array([0.27070, 0.30552, 0.18297, 0.10986, 0.06810, 0.04224]))


# --- least-squares fit of exp(-gamma r) by Gaussians ---------------------------------------

def _radial_grid(n=2000, rm=0.5):
    """Gauss-Legendre points on [0, inf) (r = rm (1 + x)/(1 - x)) and weights."""
    x, w = np.polynomial.legendre.leggauss(n)
    return rm * (1 + x) / (1 - x), w * 2 * rm / (1 - x) ** 2


def _fit_objective(log_a, gamma, k, r, w):
    """log of the weighted error int W (f - sum c g)^2 at the optimal c (weighted linear
    least squares) and its gradient in log a_i (variable projection). The residual is
    integrated pointwise (no cancellation); the logarithm makes the convergence relative."""
    a = np.exp(log_a)
    sw = np.sqrt(w * r * r * np.exp(-k * r * r))   # sqrt of weight times quadrature
    g = np.exp(-np.outer(r * r, a))
    c = np.linalg.lstsq(sw[:, None] * g, sw * np.exp(-gamma * r), rcond=None)[0]
    d = sw * (np.exp(-gamma * r) - g @ c)
    err = d @ d
    grad = 2 * a * c * ((d * sw * r * r) @ g)       # d d/d a_i = c_i r^2 g_i
    return np.log(err), grad / err, c


def stg_gtg_fit(gamma=1.0, n=6):
    """Least-squares fit exp(-gamma r) ~ sum_i c_i exp(-a_i r^2) with the weight
    r^2 exp(-(pi gamma^2)^(1/3) r^2) (Tew and Klopper; Molpro's STG-nG).

    Args:
        gamma: Slater exponent in 1/bohr.
        n: Number of Gaussians.

    Returns:
        (exponents a_i ascending, coefficients c_i).
    """
    from scipy.optimize import minimize
    k = (np.pi * gamma ** 2) ** (1.0 / 3.0)
    r, w = _radial_grid()
    starts = [np.geomspace(0.2, 200.0, n) * gamma ** p for p in (1.0, 4.0 / 3.0, 2.0)]
    if n == 6:
        starts += [_REF6[0] * gamma ** p for p in (1.0, 4.0 / 3.0, 2.0)]
    best = None
    for a0 in starts:
        res = minimize(lambda x: _fit_objective(x, gamma, k, r, w)[:2], np.log(a0), jac=True,
                       method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    log_a = np.sort(best.x)
    return np.exp(log_a), _fit_objective(log_a, gamma, k, r, w)[2]


# --- geminal and operators -----------------------------------------------------------------

@dataclass
class Geminal:
    """Correlation factor f12 = -exp(-gamma r12)/gamma, exact (exponents None) or as the
    Gaussian expansion exp(-gamma r) ~ sum_i coefficients_i exp(-exponents_i r^2)."""
    gamma: float = 1.0
    exponents: np.ndarray = None
    coefficients: np.ndarray = None

    @classmethod
    def make(cls, gamma=1.0, fit=6, exponents=None, coefficients=None):
        """Geminal from options.

        Args:
            gamma: Slater exponent in 1/bohr (>= 1e-3).
            fit: Number of fitted Gaussians, or "exact" for the Slater function itself.
            exponents: Explicit Gaussian exponents (with coefficients; override fit).
            coefficients: Explicit coefficients of exp(-gamma r) ~ sum c_i exp(-a_i r^2).
        """
        if gamma < 1e-3:
            raise ValueError(f"geminal exponent gamma = {gamma} too small (>= 1e-3)")
        if (exponents is None) != (coefficients is None):
            raise ValueError("give both geminal exponents and coefficients")
        if exponents is not None:
            a, c = np.asarray(exponents, float), np.asarray(coefficients, float)
            if a.shape != c.shape or a.ndim != 1 or np.any(a <= 0):
                raise ValueError("geminal exponents (> 0) and coefficients: equal lengths")
            return cls(gamma, a, c)
        if fit == "exact":
            return cls(gamma)
        a, c = stg_gtg_fit(gamma, int(fit))
        return cls(gamma, a, c)

    @property
    def exact(self):
        return self.exponents is None

    def describe(self):
        """One line for the output."""
        if self.exact:
            return f"Slater geminal, gamma = {self.gamma:g} (exact integrals)"
        return (f"Slater geminal, gamma = {self.gamma:g}, fitted by "
                f"{len(self.exponents)} Gaussians (STG-{len(self.exponents)}G)")

    def operators(self):
        """The F12 operators (see f12_operators)."""
        g = self.gamma
        op = _core.TwoBodyOperator
        if self.exact:
            return {
                "coulomb": op("coulomb"),
                "f12": op("stg", g, -1.0 / g),
                "f12/r12": op("stg_x_coulomb", g, -1.0 / g),
                "f12^2": op("stg", 2.0 * g, 1.0 / g ** 2),
                "(grad f12)^2": op("stg", 2.0 * g, 1.0),
            }
        gtg = self.gtg()
        return {
            "coulomb": op("coulomb"),
            "f12": op("cgtg", 0.0, -1.0 / g, gtg),
            "f12/r12": op("cgtg_x_coulomb", 0.0, -1.0 / g, gtg),
            "f12^2": op("cgtg", 0.0, 1.0 / g ** 2, gtg_product(gtg, gtg)),
            "(grad f12)^2": op("delcgtg2", 0.0, 1.0 / g ** 2, gtg),
        }

    def gtg(self):
        """Gaussian expansion [(a_i, c_i)] of exp(-gamma r)."""
        return list(zip(self.exponents.tolist(), self.coefficients.tolist()))


def gtg_product(g1, g2):
    """Expansion [(a, c)] of the product of two Gaussian expansions; for g1 == g2 the
    symmetric terms are combined (factor 2)."""
    if g1 == g2:
        return [(a1 + a2, c1 * c2 * (1 if i == j else 2))
                for i, (a1, c1) in enumerate(g1) for j, (a2, c2) in enumerate(g1) if j <= i]
    return [(a1 + a2, c1 * c2) for a1, c1 in g1 for a2, c2 in g2]


def f12_operators(gamma=1.0, fit=6, exponents=None, coefficients=None):
    """The F12 operators for a geminal (arguments as Geminal.make).

    Returns:
        dict name -> TwoBodyOperator: coulomb 1/r12, f12, f12/r12, f12^2 and
        (grad_1 f12)^2 (exactly exp(-2 gamma r12) for the Slater function).
    """
    return Geminal.make(gamma, fit, exponents, coefficients).operators()


# --- extended geminal (F12XG) --------------------------------------------------------------

class XGGeminal:
    """Extended geminal of F12XG with basis-function groups G (0-based): one Slater
    geminal f_GH = -exp(-gamma_GH r12)/gamma_GH per group pair, gamma_GH = (gt_G + gt_H)/2,
    each fitted individually (or exact). With the group-masked orbitals phi_i^G::

        g_ij(1, 2) = sum_GH f_GH(r12) phi_i^G(1) phi_j^H(2)

    Args:
        gammas: Group exponents gt_G in 1/bohr (>= 1e-3).
        fit: Number of fitted Gaussians, or "exact" (as Geminal.make).
    """

    def __init__(self, gammas, fit=6):
        self.gammas = tuple(float(g) for g in gammas)
        if not self.gammas:
            raise ValueError("no group exponents (gammas) given")
        n = len(self.gammas)
        #: group pairs (G, H), G <= H
        self.pairs = [(G, H) for G in range(n) for H in range(G, n)]
        fits = {}  # one fit per distinct exponent (equal gammas give identical geminals)
        self.geminals = {}
        for G, H in self.pairs:
            g = 0.5 * (self.gammas[G] + self.gammas[H])
            if g not in fits:
                fits[g] = Geminal.make(g, fit)
            self.geminals[(G, H)] = fits[g]

    @property
    def exact(self):
        return self.geminals[self.pairs[0]].exact

    def geminal(self, G, H):
        """Geminal f_GH of the group pair (either order)."""
        return self.geminals[(min(G, H), max(G, H))]

    def operators(self, G, H):
        """Operators f12 and f12/r12 of the group pair (G, H)."""
        ops = self.geminal(G, H).operators()
        return {name: ops[name] for name in ("f12", "f12/r12")}

    def product_operators(self, bra, ket):
        """Operators with the geminals of two group pairs, bra = (G', H'), ket = (G, H):
        'f12 x f12' = f_G'H' f_GH and 'grad f12 . grad f12' = (grad_1 f_G'H').(grad_1 f_GH)
        (for equal geminals the operators f12^2 and (grad f12)^2 of the geminal). Exact:
        exp(-(g_a + g_b) r12) with the factors 1/(g_a g_b) and 1; fitted: libint
        delcgtga_delcgtgb with 1/(g_a g_b)."""
        a, b = self.geminal(*bra), self.geminal(*ket)
        op = _core.TwoBodyOperator
        scale = 1.0 / (a.gamma * b.gamma)
        if a.exact:
            return {"f12 x f12": op("stg", a.gamma + b.gamma, scale),
                    "grad f12 . grad f12": op("stg", a.gamma + b.gamma, 1.0)}
        ga, gb = a.gtg(), b.gtg()
        grad = (op("delcgtg2", 0.0, scale, ga) if ga == gb else
                op("delcgtga_delcgtgb", 0.0, scale, ga, gb))
        return {"f12 x f12": op("cgtg", 0.0, scale, gtg_product(ga, gb)),
                "grad f12 . grad f12": grad}

    def describe(self):
        """Lines for the output: group exponents and the geminal of each group pair."""
        lines = ["group exponents " + " ".join(f"{g:g}" for g in self.gammas)]
        lines += [f"pair {G + 1}{H + 1}: " + self.geminals[(G, H)].describe()
                  for G, H in self.pairs]
        return lines
