"""Integrals vs. analytic results for single normalized Gaussian primitives."""

from math import exp, pi, sqrt

import numpy as np
import pytest

import gemqc

DATA = __import__("pathlib").Path(__file__).parent / "data"


def one_atom(z=1, r=(0.0, 0.0, 0.0)):
    mol = gemqc.Molecule("atom")
    mol.add_atom(z, list(r))
    return mol


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_single_primitive(l):
    a = 0.8
    mol = one_atom()
    basis = gemqc.BasisSet(mol, {1: [gemqc.ShellDef(l, True, [a], [1.0])]})
    S, T = gemqc.overlap(basis), gemqc.kinetic(basis)
    assert np.allclose(S, np.eye(2 * l + 1), atol=1e-13)
    assert np.allclose(T, (2 * l + 3) * a / 2 * np.eye(2 * l + 1), atol=1e-12)
    if l == 0:  # <s|-1/r|s> = -2 sqrt(2a/pi)
        V = gemqc.nuclear_attraction(basis, mol)
        assert V[0, 0] == pytest.approx(-2 * sqrt(2 * a / pi), rel=1e-12)


def test_two_center_overlap_and_dipole():
    a, b, R = 0.7, 1.3, 1.1
    mol = gemqc.Molecule()
    mol.add_atom(1, [0.0, 0.0, 0.0])
    mol.add_atom(2, [0.0, 0.0, R])
    lib = {1: [gemqc.ShellDef(0, True, [a], [1.0])],
           2: [gemqc.ShellDef(0, True, [b], [1.0])]}
    basis = gemqc.BasisSet(mol, lib)
    S = gemqc.overlap(basis)
    ref = (2 * sqrt(a * b) / (a + b)) ** 1.5 * exp(-a * b / (a + b) * R**2)
    assert S[0, 1] == pytest.approx(ref, rel=1e-12)
    X, Y, Z = gemqc.dipole(basis)
    assert Z[1, 1] == pytest.approx(R, rel=1e-12)          # <z> of function at R
    assert Z[0, 1] == pytest.approx(b * R / (a + b) * ref)  # Gaussian product centre
    assert abs(X[0, 1]) < 1e-14


def test_set_centers():
    a = 0.5
    mol = one_atom()
    basis = gemqc.BasisSet(mol, {1: [gemqc.ShellDef(0, True, [a], [1.0])]})
    mol.set_position(0, [0.0, 0.0, 2.0])
    basis.set_centers(mol)
    assert gemqc.dipole(basis)[2][0, 0] == pytest.approx(2.0)


def test_am_limit():
    lmax = gemqc.libint_info()["max_am"]
    with pytest.raises(ValueError):
        gemqc.BasisSet(one_atom(), {1: [gemqc.ShellDef(lmax + 1, True, [1.0], [1.0])]})


def test_water_def2svp():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    basis = gemqc.build_basis(mol, "def2-SVP")
    assert basis.nbf == 24 and basis.nshells == 12 and basis.max_l == 2
    S = gemqc.overlap(basis)
    T = gemqc.kinetic(basis)
    V = gemqc.nuclear_attraction(basis, mol)
    for M in (S, T, V):
        assert np.allclose(M, M.T, atol=1e-14)
    assert np.allclose(np.diag(S), 1.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(T) > 0)
    assert np.all(np.diag(V) < 0)


def test_mixed_basis():
    """<A|op|B> equals the off-diagonal block of the combined basis A+B."""
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    svp = gemqc.load_basis("def2-SVP")
    # basis A: more primitives (7) and higher l (f) than def2-SVP -> tests engine sizing
    exps = [30.0, 12.0, 5.0, 2.0, 0.8, 0.3, 0.1]
    lib_a = {z: [gemqc.ShellDef(0, True, exps, [0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05]),
                 gemqc.ShellDef(3, True, [0.9], [1.0])] for z in (1, 8)}
    a = gemqc.BasisSet(mol, lib_a, "A")
    b = gemqc.build_basis(mol, "def2-SVP")
    ab = gemqc.BasisSet(mol, {z: lib_a[z] + svp[z] for z in (1, 8)}, "A+B")

    # functions of A and B per atom in ab: map to positions of the A and B blocks
    idx_a, idx_b = [], []
    for i in range(ab.nshells):
        sh, off = ab.shell(i), ab.shell_offsets[i]
        n = 2 * sh["l"] + 1
        is_a = sh["l"] == 3 or len(sh["exponents"]) == 7
        (idx_a if is_a else idx_b).extend(range(off, off + n))
    assert len(idx_a) == a.nbf and len(idx_b) == b.nbf

    def block(m):
        return m[np.ix_(idx_a, idx_b)]

    assert np.allclose(gemqc.overlap(a, b), block(gemqc.overlap(ab)), atol=1e-13)
    assert np.allclose(gemqc.kinetic(a, b), block(gemqc.kinetic(ab)), atol=1e-12)
    assert np.allclose(gemqc.nuclear_attraction(a, mol, b),
                       block(gemqc.nuclear_attraction(ab, mol)), atol=1e-12)
    for d_ab, d_full in zip(gemqc.dipole(a, basis2=b), gemqc.dipole(ab)):
        assert np.allclose(d_ab, block(d_full), atol=1e-13)
    # <B|A> = <A|B>^T, and same basis twice gives the square matrix
    assert np.allclose(gemqc.overlap(b, a), gemqc.overlap(a, b).T, atol=1e-14)
    assert np.allclose(gemqc.overlap(b, b), gemqc.overlap(b), atol=1e-14)
    assert gemqc.overlap(a, b).shape == (a.nbf, b.nbf)
