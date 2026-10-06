"""CABS for F12 methods (gemqc.cabs, task mp2-f12 so far): CABS orbitals, Fock matrices in
the RI basis, CABS singles against the coupled equations and Molpro, auxiliary-basis
options of the task."""

from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.cabs import build_ri_space, cabs_singles
from gemqc.correlation import orbital_spaces
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"


def _water(basis="cc-pVDZ"):
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    bs = gemqc.build_basis(mol, basis)
    o = gemqc.ScfOptions()
    o.convergence = 1e-10
    return gemqc.rhf(mol, bs, o, gemqc.sad_guess(mol, bs))


def _space(wf, cabs="aug-cc-pVDZ-CABS", jk=None, frozen="off"):
    sp = orbital_spaces(wf, frozen=frozen, deleted="off")
    a = sp.alpha
    return sp, build_ri_space(wf.molecule, wf.basis.name, a.C, a.nfrozen + a.nocc, cabs,
                              1e-8, jk)


def test_cabs_orbitals(quiet):
    """CABS orthonormal and orthogonal to the orbital basis; exact Fock matrix diagonal
    in the canonical MOs (orbital energies); f = (h + 2J) - K."""
    wf = _water()
    sp, ri = _space(wf)
    S = np.asarray(_core.overlap(ri.basis))
    G = ri.C.T @ S @ ri.C
    assert np.abs(G - np.eye(G.shape[0])).max() < 1e-10
    assert ri.ncabs == len(ri.aux_idx) - ri.dropped and ri.nmo == wf.basis.nbf
    n = ri.nmo
    assert np.abs(ri.fock[:n, :n] - np.diag(sp.alpha.eps)).max() < 1e-8
    assert np.abs(ri.fock - (ri.hj - ri.exchange)).max() < 1e-14
    assert np.abs(ri.fock - ri.fock.T).max() < 1e-10
    assert np.abs(ri.fock[:5, n:]).max() > 1e-2  # occupied-CABS coupling (the singles)


def _coupled_singles(F, occ, vir):
    """CABS singles from the coupled equations f_Ai + sum_B f_AB t_iB - sum_j t_jA f_ji = 0,
    E = 2 sum f_iA t_iA (no semicanonical transformation)."""
    Foo, Fvv, Fov = F[np.ix_(occ, occ)], F[np.ix_(vir, vir)], F[np.ix_(occ, vir)]
    no, nv = len(occ), len(vir)
    A = np.kron(np.eye(no), Fvv) - np.kron(Foo.T, np.eye(nv))  # t[i, A] row-major
    t = np.linalg.solve(A, -Fov.reshape(-1))
    return 2.0 * Fov.reshape(-1) @ t


def test_cabs_singles(quiet):
    """Semicanonical CABS singles = coupled equations (all electrons and without the core);
    DF Fock matrices close to exact; the orbital basis as CABS gives no CABS and zero
    singles (Brillouin)."""
    wf = _water()
    sp, ri = _space(wf)
    e_all, e_i = cabs_singles(ri)
    vir = ri.virtual
    assert e_all == pytest.approx(_coupled_singles(ri.fock, ri.occupied, vir), abs=1e-12)
    assert e_all == pytest.approx(e_i.sum(), abs=1e-15) and e_all < -1e-4
    e_val = cabs_singles(ri, ri.occupied[1:])[0]
    assert e_val == pytest.approx(_coupled_singles(ri.fock, ri.occupied[1:], vir), abs=1e-12)
    assert e_all < e_val < 0.0
    _, ri_df = _space(wf, jk="cc-pVTZ-JK")
    assert cabs_singles(ri_df)[0] == pytest.approx(e_all, abs=2e-5)
    _, ri0 = _space(wf, cabs="cc-pVDZ")
    assert ri0.ncabs == 0 and ri0.dropped == wf.basis.nbf
    assert abs(cabs_singles(ri0)[0]) < 1e-14


def _run(tmp_path, tasks, system):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "job.yaml"
    inp.write_text(f"systems:\n  {system}\ntasks:\n" + tasks)
    return [r for _, r in gemqc.run(inp, quiet=True)]


def test_task_auxiliary_bases(tmp_path):
    """df-mp2-f12: default CABS file, 'as before' and task options for jk, c and cabs
    auxiliary basis sets; the MP2 part equals df-mp2; the frozen core lowers the CABS
    singles only without all_electron_cabs."""
    res = _run(tmp_path,
               "  - {system: w, task: df-rhf, convergence: 1e-9}\n"
               "  - {task: df-mp2}\n"
               "  - {task: df-mp2-f12}\n"
               "  - {task: df-mp2-f12, all_electron_cabs: false}\n"
               "  - {task: df-mp2-f12, cabs_auxbasis: cc-pVTZ-JK, c_auxbasis: cc-pVTZ-C,"
               " jk_auxbasis: cc-pVQZ-JK}\n"
               "  - {task: df-mp2-f12, c_auxbasis: aug-cc-pVDZ-C}\n",
               "w: {coordinates: h2o.xyz, basis: aug-cc-pVDZ}")
    scf_, mp2, f12, val, other, exact = res
    assert f12["cabs_auxbasis"] == "aug-cc-pVDZ-CABS" and val["cabs_auxbasis"] == "aug-cc-pVDZ-CABS"
    assert f12["density_fitting"]["c_auxbasis"] == "aug-cc-pVDZ-C"
    assert f12["mp2_correlation_energy"] == pytest.approx(mp2["correlation_energy"], abs=1e-12)
    assert f12["e_cabs_singles"] < val["e_cabs_singles"] < 0.0
    assert f12["energy"] == pytest.approx(scf_["energy"] + f12["e_cabs_singles"]
                                          + f12["mp2_correlation_energy"] + f12["e_f12"],
                                          abs=1e-12)
    assert other["cabs_auxbasis"] == "cc-pVTZ-JK"
    assert other["density_fitting"]["c_auxbasis"] == "cc-pVTZ-C"
    assert other["e_cabs_singles"] != pytest.approx(f12["e_cabs_singles"], abs=1e-6)
    assert exact["cabs_auxbasis"] == "cc-pVTZ-JK"  # as before
    assert exact["method"] == "DF-MP2-F12"


def test_molpro_cabs_singles(tmp_path):
    """Water aug-cc-pVTZ, DF-RHF and CABS singles with RI = JK basis AVTZ/JKFIT against
    Molpro (reference_f12/h2o-dfmp2-testcabs-molpro): frozen core excluded, all electrons.
    aug-cc-pVTZ-JK is generated by lib/basis/augment_jk.py."""
    system = ("w:\n    basis: aug-cc-pVTZ\n    jk_auxbasis: aug-cc-pVTZ-JK\n"
              "    symmetry: nosym\n    geometry: |\n"
              "      O     0.0000000    0.0000000    0.3935881\n"
              "      H     0.7571222    0.0000000   -0.1967941\n"
              "      H    -0.7571222    0.0000000   -0.1967941")
    res = _run(tmp_path,
               "  - {system: w, task: df-rhf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12, cabs_auxbasis: aug-cc-pVTZ-JK, all_electron_cabs: false}\n"
               "  - {task: df-mp2-f12, frozen: off}\n", system)
    scf_, fc, ae = res
    assert scf_["energy"] == pytest.approx(-76.060373948805, abs=1e-8)
    assert fc["e_cabs_singles"] == pytest.approx(-0.004751975710, abs=1e-8)
    assert ae["e_cabs_singles"] == pytest.approx(-0.004881362855, abs=1e-8)
    assert fc["mp2_correlation_energy"] == pytest.approx(-0.268560636102, abs=1e-8)
    assert ae["mp2_correlation_energy"] == pytest.approx(-0.283714428370, abs=1e-8)
