"""Second-order SCF solver (augmented Hessian Newton, trust region)."""

import json
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc.symmetry import scf_symmetry, symmetrize_molecule
from test_qn import quiet, solve, water  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("xyz, e_ref", [("h2o.xyz", -75.960841436478),
                                        ("butane.xyz", -157.184381601459)])
def test_newton_reference_energies(xyz, e_ref):
    mol = gemqc.read_xyz(DATA / xyz)
    sym = symmetrize_molecule(mol)
    bs = gemqc.build_basis(mol, "def2-SVP")
    wf = solve(mol, bs, scf_symmetry(bs, sym), "newton", convergence=1e-8,
               energy_convergence=1e-10)
    assert wf.converged and wf.energy == pytest.approx(e_ref, abs=1e-9)
    S = gemqc.overlap(bs)  # canonical orbitals at the end
    assert np.allclose(wf.C.T @ S @ wf.C, np.eye(bs.nbf), atol=1e-10)
    assert np.all(np.diff(wf.eps) >= -1e-12)


@pytest.mark.parametrize("scale, e_min", [(2.5, -75.4068823493), (3.0, -75.3682896033)])
def test_newton_stretched(scale, e_min):
    """Minimum (DIIS ends on a saddle point) in fewer iterations than QN;
    quadratic convergence at the end."""
    mol, bs, ss = water(scale)
    wf = solve(mol, bs, ss, "newton", convergence=1e-8, energy_convergence=1e-10)
    qn = solve(mol, bs, ss, "qn", convergence=1e-8, energy_convergence=1e-10)
    assert wf.converged and wf.energy == pytest.approx(e_min, abs=1e-9)
    assert wf.iterations < qn.iterations and wf.iterations <= 12


def test_newton_c1_and_small_trust_radius():
    mol, bs, ss = water(1.5)
    ref = solve(mol, bs, ss, "diis", convergence=1e-8).energy
    wf = solve(mol, bs, None, "newton", convergence=1e-8)
    assert wf.converged and wf.energy == pytest.approx(ref, abs=1e-9)
    wf = solve(mol, bs, ss, "newton", convergence=1e-8, trust_radius=0.05, qn_start=2,
               qn_stable=1, max_iterations=200)
    assert wf.converged and wf.energy == pytest.approx(ref, abs=1e-9)


def test_driver_newton(tmp_path):
    r, th = 0.9572 * 2.5, np.radians(104.5) / 2
    (tmp_path / "w.xyz").write_text(f"3\n\nO 0 0 0\nH 0 {r * np.sin(th)} {r * np.cos(th)}\n"
                                    f"H 0 {-r * np.sin(th)} {r * np.cos(th)}\n")
    inp = tmp_path / "n.yaml"
    inp.write_text("systems:\n  w: {coordinates: w.xyz, basis: def2-SVP}\n"
                   "tasks:\n  - {task: scf, system: w, guess: sad, solver: newton,"
                   " save: false}\n")
    gemqc.run(inp, quiet=True)
    res = json.loads(inp.with_suffix(".json").read_text())["tasks"][0]["results"]
    assert res["energy"] == pytest.approx(-75.4068823493, abs=1e-8)
