"""High-spin ROHF (general open-shell form): solvers, symmetry, occupation,
densities, task synonyms scf/rhf/rohf, restart."""

import json
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.symmetry import scf_symmetry, symmetrize_molecule
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"
E_NH2 = -55.511421521766  # Molpro, def2-SVP
E_CH2 = -38.888300592063


def radical(name, mult, symmetry=True, **occ):
    mol = gemqc.read_xyz(DATA / f"{name}.xyz")
    mol.multiplicity = mult
    sym = symmetrize_molecule(mol, request="auto" if symmetry else "nosym")
    bs = gemqc.build_basis(mol, "def2-SVP")
    return mol, bs, scf_symmetry(bs, sym, **occ) or _core.ScfSymmetry()


def run_rohf(mol, bs, ss, solver="diis", start=None, **kw):
    o = gemqc.ScfOptions()
    o.solver, o.convergence, o.energy_convergence = solver, 1e-8, 1e-10
    for k, v in kw.items():
        setattr(o, k, v)
    if start is not None:
        return gemqc.rohf(mol, bs, o, [], ss, start)
    return gemqc.rohf(mol, bs, o, [gemqc.sad_guess(mol, bs)], ss)


@pytest.mark.parametrize("solver", ["diis", "qn"])
@pytest.mark.parametrize("symmetry", [True, False])
def test_rohf_solvers(solver, symmetry):
    mol, bs, ss = radical("nh2", 2, symmetry)
    wf = run_rohf(mol, bs, ss, solver)
    assert wf.converged and wf.energy == pytest.approx(E_NH2, abs=1e-9)
    S = gemqc.overlap(bs)
    assert np.allclose(wf.C.T @ S @ wf.C, np.eye(bs.nbf), atol=1e-10)
    assert sorted(wf.occ)[-5:] == [1, 2, 2, 2, 2]
    occ = list(wf.occ)  # order: closed, open, virtual (open shells on top)
    assert occ == sorted(occ, reverse=True) and occ[4] == 1.0
    assert np.sum(wf.D * S) == pytest.approx(9.0) and np.sum(wf.D_spin * S) == pytest.approx(1.0)
    assert wf.s2 == wf.s2_exact == 0.75
    if symmetry:  # Molpro: alpha 3 1 1 0, beta 3 1 0 0
        assert list(wf.occupation) == [3, 1, 0, 0] and list(wf.occupation_open) == [0, 0, 1, 0]


def test_rohf_fixed_occupation_and_start():
    """Excited open-shell configuration of NH2 (open shell in A1 instead of
    B2) by fixed occupation; QN from given start orbitals keeps it."""
    mol, bs, ss = radical("nh2", 2, occupation=[2, 1, 1, 0], occupation_open=[1, 0, 0, 0])
    wf = run_rohf(mol, bs, ss)
    assert wf.converged and wf.energy > E_NH2
    assert list(wf.occupation) == [2, 1, 1, 0] and list(wf.occupation_open) == [1, 0, 0, 0]
    mol, bs, ss = radical("nh2", 2)
    again = run_rohf(mol, bs, ss, "qn", start=wf)
    assert again.converged and again.energy == pytest.approx(wf.energy, abs=1e-8)
    assert list(again.occupation_open) == [1, 0, 0, 0]


def test_rohf_triplet_c1_and_errors():
    mol, bs, ss = radical("ch2", 3, symmetry=False)
    wf = run_rohf(mol, bs, ss, "qn")
    assert wf.converged and wf.energy == pytest.approx(E_CH2, abs=1e-9) and wf.s2 == 2.0
    with pytest.raises(ValueError, match="Newton solver is not available for ROHF"):
        run_rohf(mol, bs, ss, "newton")
    mol.multiplicity = 1
    with pytest.raises(ValueError, match="needs an open-shell state"):
        run_rohf(mol, bs, ss)


def test_task_synonyms(tmp_path):
    for f in ("nh2.xyz", "h2o.xyz"):
        (tmp_path / f).write_text((DATA / f).read_text())
    inp = tmp_path / "s.yaml"
    inp.write_text("systems:\n  nh2: {coordinates: nh2.xyz, basis: def2-SVP, multiplicity: 2}\n"
                   "  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "tasks:\n"
                   "  - {task: scf, system: nh2}\n"
                   "  - {task: rhf, system: nh2, solver: newton, save: false,"
                   " occupation: {closed: [3, 1, 0, 0], open: [0, 0, 1, 0]}}\n"
                   "  - {task: rohf, system: w, save: false}\n"
                   "  - {task: rohf, system: nh2, save: false}\n")
    gemqc.run(inp, quiet=True)
    t = [x["results"] for x in json.loads(inp.with_suffix(".json").read_text())["tasks"]]
    assert [r["method"] for r in t] == ["ROHF", "ROHF", "RHF", "ROHF"]
    assert t[0]["energy"] == pytest.approx(E_NH2, abs=1e-8)
    assert t[0]["open_shells_per_irrep"] == [0, 0, 1, 0]
    assert t[1]["closed_shells_per_irrep"] == [3, 1, 0, 0]
    assert t[3]["iterations"] <= 3  # restart from the stored ROHF
    inp.write_text("systems:\n  nh2: {coordinates: nh2.xyz, basis: def2-SVP, multiplicity: 2}\n"
                   "tasks:\n  - {task: rohf, system: nh2, occupation: {alpha: [3, 1, 1, 0]}}\n")
    with pytest.raises(gemqc.InputError, match="closed: "):
        gemqc.run(inp, quiet=True)
