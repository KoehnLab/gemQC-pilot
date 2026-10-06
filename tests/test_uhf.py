"""Unrestricted Hartree-Fock: energies, <S^2>, natural orbitals, solvers,
driver and restart."""

import json
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.symmetry import scf_symmetry, symmetrize_molecule
from test_qn import quiet, water  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"
E_WATER_SVP = -75.960841436478
E_CATION = -75.562402786  # H2O+ (h2o.xyz), doublet, def2-SVP


def cation(symmetry=True):
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    mol.charge, mol.multiplicity = 1, 2
    sym = symmetrize_molecule(mol)
    bs = gemqc.build_basis(mol, "def2-SVP")
    return mol, bs, scf_symmetry(bs, sym) if symmetry else None


def run_uhf(mol, bs, ss, solver="diis", **kw):
    o = gemqc.ScfOptions()
    o.solver, o.convergence, o.energy_convergence = solver, 1e-8, 1e-10
    for k, v in kw.items():
        setattr(o, k, v)
    return gemqc.uhf(mol, bs, o, [gemqc.sad_guess(mol, bs)], ss or _core.ScfSymmetry())


def test_uhf_closed_shell_equals_rhf():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    bs = gemqc.build_basis(mol, "def2-SVP")
    wf = run_uhf(mol, bs, None)
    assert wf.method == "UHF" and wf.energy == pytest.approx(E_WATER_SVP, abs=1e-9)
    assert abs(wf.s2) < 1e-10 and np.allclose(wf.eps, wf.eps_b, atol=1e-7)


@pytest.mark.parametrize("solver", ["diis", "qn", "newton"])
@pytest.mark.parametrize("symmetry", [True, False])
def test_uhf_doublet(solver, symmetry):
    mol, bs, ss = cation(symmetry)
    wf = run_uhf(mol, bs, ss, solver)
    assert wf.converged and wf.energy == pytest.approx(E_CATION, abs=1e-8)
    assert wf.s2 == pytest.approx(0.7562874, abs=1e-6) and wf.s2_exact == 0.75
    assert int(wf.occ.sum()) == 5 and int(wf.occ_b.sum()) == 4
    S = gemqc.overlap(bs)
    # densities, natural orbitals: sum 9 electrons, one singly occupied orbital
    assert np.sum(wf.D * S) == pytest.approx(9.0) and np.sum(wf.D_spin * S) == pytest.approx(1.0)
    assert wf.occ_nat.sum() == pytest.approx(9.0) and np.all(np.diff(wf.occ_nat) <= 1e-12)
    assert np.sum(np.abs(wf.occ_nat - 1.0) < 1e-6) == 1
    assert np.allclose(wf.C_nat.T @ S @ wf.C_nat, np.eye(bs.nbf), atol=1e-10)
    if solver == "newton":
        assert wf.iterations <= 9


def test_driver_uhf(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "u.yaml"
    inp.write_text("systems:\n  cat: {coordinates: h2o.xyz, basis: def2-SVP, charge: 1,"
                   " multiplicity: 2}\n"
                   "tasks:\n  - {task: uhf, system: cat, convergence: 1e-8,"
                   " orbitals: long}\n"
                   "  - {task: uhf, system: cat, occupation: {alpha: [3, 1, 1, 0],"
                   " beta: [3, 1, 0, 0]}, save: false}\n")
    gemqc.run(inp, quiet=True)
    t = json.loads(inp.with_suffix(".json").read_text())["tasks"]
    r = t[0]["results"]
    assert r["method"] == "UHF" and r["energy"] == pytest.approx(E_CATION, abs=1e-8)
    assert r["alpha_orbitals_per_irrep"] == [3, 1, 1, 0]
    assert r["beta_orbitals_per_irrep"] == [3, 1, 0, 0]  # xz plane kept (as Molpro)
    assert sum(r["natural_occupations"]) == pytest.approx(9.0)
    assert len(r["orbital_labels_beta"]) == len(r["natural_orbital_labels"]) == 24
    assert t[1]["results"]["iterations"] <= 3  # restart from the stored alpha/beta densities
    inp.write_text("systems:\n  cat: {coordinates: h2o.xyz, basis: def2-SVP, charge: 1,"
                   " multiplicity: 2}\n"
                   "tasks:\n  - {task: uhf, system: cat, occupation: [3, 1, 1, 0]}\n")
    with pytest.raises(gemqc.InputError, match="alpha: "):
        gemqc.run(inp, quiet=True)


def test_uhf_multiplicity_checks():
    mol, bs, _ = cation(False)
    mol.multiplicity = 1  # 9 electrons
    with pytest.raises(ValueError, match="multiplicity 1 impossible"):
        run_uhf(mol, bs, None)
