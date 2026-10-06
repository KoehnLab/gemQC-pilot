"""Properties: multipole moments, spin density on HDF5."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import gemqc
from gemqc import _core

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def quiet():
    _core.set_output_enabled(False)
    yield
    _core.set_output_enabled(True)


def run(tmp_path, text, name="p"):
    for f in ("h2o.xyz", "ch2.xyz"):
        (tmp_path / f).write_text((DATA / f).read_text())
    inp = tmp_path / f"{name}.yaml"
    inp.write_text(text)
    return [r for _, r in gemqc.run(inp, quiet=True)]


def test_scf_moments_and_h5(tmp_path):
    """RHF water dipole vs Molpro 2025.5 (def2-SVP): -0.84258218 au; origin, traceless
    quadrupole, translation invariance; ROHF spin density (trace 2S) on HDF5."""
    rhf, rohf, uhf, off = run(
        tmp_path, "systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
        "  c: {coordinates: ch2.xyz, basis: def2-SVP, multiplicity: 3}\n"
        "tasks:\n  - {system: w, task: scf, convergence: 1e-9}\n  - {system: c, task: rohf}\n"
        "  - {task: uhf}\n  - {system: w, task: scf, properties: false}\n")
    assert rhf["dipole"][2] == pytest.approx(-0.84258218, abs=2e-8)
    assert rhf["dipole_debye"] == pytest.approx(2.14163028, abs=1e-6)
    q = rhf["quadrupole"]
    assert abs(q[0] + q[1] + q[2]) < 1e-12 and "dipole" not in off
    mol = rhf["wavefunction"].molecule
    shifted = gemqc.Molecule("w")
    for z, r in zip(mol.atomic_numbers, np.asarray(mol.coordinates)):
        shifted.add_atom(int(z), list(r + [0.3, -0.2, 0.5]))
    from gemqc.basislib import build_basis
    from gemqc.properties import Multipoles
    mp = Multipoles(shifted, build_basis(shifted, "def2-SVP"))
    _core.set_output_enabled(False)  # enabled again by gemqc.run
    wf = gemqc.rhf(shifted, build_basis(shifted, "def2-SVP"))
    m = mp.moments(np.asarray(wf.D))
    assert np.allclose(m["dipole"], rhf["dipole"], atol=1e-6)
    assert np.allclose(m["quadrupole"], q, atol=1e-5)  # SCF convergence of wf
    S = np.asarray(_core.overlap(rohf["wavefunction"].basis))
    with h5py.File(tmp_path / "p.h5") as f:
        g = f["c/scf"]
        assert np.sum(g["D_spin"][()] * S) == pytest.approx(2.0)
        assert np.allclose(g.attrs["dipole"], rohf["dipole"])
        assert np.sum(f["c/uhf/D_spin"][()] * S) == pytest.approx(2.0)
