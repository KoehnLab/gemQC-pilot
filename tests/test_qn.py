"""Quasi-Newton SCF solver (L-BFGS on orbital rotations)."""

from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.symmetry import scf_symmetry, symmetrize_molecule

DATA = Path(__file__).parent / "data"
A = 1 / 0.529177210903


@pytest.fixture(autouse=True)
def quiet():
    _core.set_output_enabled(False)
    yield
    _core.set_output_enabled(True)


def water(scale=1.0, basis="def2-SVP"):
    r, th = 0.9572 * scale * A, np.radians(104.5) / 2
    mol = gemqc.Molecule("water")
    for z, p in [(8, [0, 0, 0]), (1, [0, r * np.sin(th), r * np.cos(th)]),
                 (1, [0, -r * np.sin(th), r * np.cos(th)])]:
        mol.add_atom(z, p)
    sym = symmetrize_molecule(mol)
    bs = gemqc.build_basis(mol, basis)
    return mol, bs, scf_symmetry(bs, sym)


def solve(mol, bs, ss, solver, guess="sad", **kw):
    o = gemqc.ScfOptions()
    o.solver = solver
    for k, v in kw.items():
        setattr(o, k, v)
    D0 = gemqc.sad_guess(mol, bs) if guess == "sad" else None
    return gemqc.rhf(mol, bs, o, D0, ss) if ss else gemqc.rhf(mol, bs, o, D0)


@pytest.mark.parametrize("xyz, e_ref", [("h2o.xyz", -75.960841436478),
                                        ("butane.xyz", -157.184381601459)])
def test_qn_reference_energies(xyz, e_ref):
    mol = gemqc.read_xyz(DATA / xyz)
    sym = symmetrize_molecule(mol)
    bs = gemqc.build_basis(mol, "def2-SVP")
    wf = solve(mol, bs, scf_symmetry(bs, sym), "qn", convergence=1e-8, energy_convergence=1e-10)
    assert wf.converged and wf.energy == pytest.approx(e_ref, abs=1e-9)
    # canonical orbitals: orthonormal, Fock matrix diagonal in the MO basis
    S = gemqc.overlap(bs)
    assert np.allclose(wf.C.T @ S @ wf.C, np.eye(bs.nbf), atol=1e-10)
    assert np.all(np.diff(wf.eps) >= -1e-12)


def test_qn_c1_matches_diis():
    mol, bs, _ = water(1.5)
    e_diis = solve(mol, bs, None, "diis").energy
    wf = solve(mol, bs, None, "qn")
    assert wf.converged and wf.energy == pytest.approx(e_diis, abs=1e-8)


def test_qn_finds_lower_minimum():
    """Stretched water: DIIS converges to a higher stationary point."""
    mol, bs, ss = water(2.5)
    e_diis = solve(mol, bs, ss, "diis").energy
    wf = solve(mol, bs, ss, "qn")
    assert wf.converged and wf.energy < e_diis - 1e-3


def test_qn_waits_for_stable_occupation():
    """Core guess: wrong initial occupation is corrected by DIIS before QN starts."""
    mol, bs, ss = water(1.5)
    wf = solve(mol, bs, ss, "qn", guess="core")
    assert wf.converged and list(wf.occupation) == [3, 1, 1, 0]
    assert wf.energy == pytest.approx(solve(mol, bs, ss, "diis").energy, abs=1e-8)


def test_qn_fixed_occupation():
    mol, bs, ss = water()
    ss.occupation = [2, 1, 2, 0]
    e_diis = solve(mol, bs, ss, "diis").energy
    wf = solve(mol, bs, ss, "qn")
    assert list(wf.occupation) == [2, 1, 2, 0]
    assert wf.energy == pytest.approx(e_diis, abs=1e-7)


def test_unknown_solver():
    mol, bs, ss = water()
    with pytest.raises(ValueError, match="unknown SCF solver"):
        solve(mol, bs, ss, "bfgs")
