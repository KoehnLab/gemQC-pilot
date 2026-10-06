"""MP2 and DF-MP2 (task mp2): brute-force spin-orbital MP2, DF against exact, storage,
symmetry, frozen core, task results and errors."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.correlation import orbital_spaces
from gemqc.mp2 import mp2_energy
from gemqc.options import InputError
from gemqc.transform import MOIntegrals
from test_correlation import scf
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"


def brute_force(sp, basis):
    """Spin-orbital MP2 over the active spaces from the full ERI tensor (+ singles)."""
    n = basis.nbf
    eri = np.asarray(_core.eri(basis)).reshape(n, n, n, n)
    a, b = sp.alpha, sp.beta
    if a is b:
        b = copy.copy(a)
    E = 0.0
    for s1, s2 in ((a, a), (b, b), (a, b)):
        X = np.einsum("pqrs,pi,qa,rj,sb->iajb", eri, s1.C[:, s1.occ], s1.C[:, s1.virt],
                      s2.C[:, s2.occ], s2.C[:, s2.virt], optimize=True)
        eo1, ev1, eo2, ev2 = s1.eps[s1.occ], s1.eps[s1.virt], s2.eps[s2.occ], s2.eps[s2.virt]
        D = (eo1[:, None, None, None] - ev1[None, :, None, None] + eo2[None, None, :, None]
             - ev2[None, None, None, :])
        if s1 is s2:
            A = X - X.transpose(0, 3, 2, 1)
            E += 0.25 * np.sum(A * A / D)
        else:
            E += np.sum(X * X / D)
    if sp.reference == "ROHF":
        for s in (a, b):
            f = s.F[s.occ, s.virt]
            E += np.sum(f * f / (s.eps[s.occ][:, None] - s.eps[s.virt][None, :]))
    return E


@pytest.mark.parametrize("name, mult, method, mode", [
    ("h2o", 1, "rhf", "freeze_first"), ("h2o", 1, "uhf", "freeze_first"),
    ("ch2", 3, "uhf", "freeze_first"), ("ch2", 3, "rohf", "freeze_first"),
    ("ch2", 3, "rohf", "diag_first")])
def test_against_brute_force(quiet, name, mult, method, mode):
    """Closed-shell, UHF and semicanonical ROHF (with singles) MP2 against the spin-orbital
    formula; memory and disk storage (small blocks) agree."""
    wf = scf(name, mult, method)
    sp = orbital_spaces(wf, frozen="auto", deleted=2, semicanonical=mode)
    fb = _core.FockBuilder(wf.basis, 1e-14)
    ints = MOIntegrals(sp, fb)
    r = mp2_energy(sp, ints, pair_energies=True)
    assert r["corr"] == pytest.approx(brute_force(sp, wf.basis), abs=1e-12)
    assert sum(e.sum() for e in r["pairs"].values()) == pytest.approx(r["corr"] - r["singles"],
                                                                      abs=1e-12)
    disk = MOIntegrals(sp, fb, mode="disk", block_bytes=2000)
    assert disk.on_disk and disk.path.exists()
    assert mp2_energy(sp, disk)["corr"] == pytest.approx(r["corr"], abs=1e-13)
    path = disk.path
    disk.close()
    assert not path.exists()
    assert (r["singles"] != 0.0) == (method == "rohf")


def test_closed_shell_equals_uhf(quiet):
    """For a closed-shell molecule UHF = RHF, and so are the MP2 components."""
    r = []
    for wf in (scf("h2o"), scf("h2o", 1, "uhf")):
        sp = orbital_spaces(wf)
        r.append(mp2_energy(sp, MOIntegrals(sp, _core.FockBuilder(wf.basis, 1e-14))))
    for k in ("os", "ss", "corr"):
        assert r[0][k] == pytest.approx(r[1][k], abs=1e-10)


def test_density_fitting(quiet):
    """DF-MP2 (def2-SVP-C) against exact MP2: RHF and ROHF within 2e-4 Eh."""
    for args in (("h2o",), ("ch2", 3, "rohf")):
        wf = scf(*args)
        sp = orbital_spaces(wf)
        aux = gemqc.build_basis(wf.molecule, "def2-SVP-C")
        e_df = mp2_energy(sp, MOIntegrals(sp, _core.DFBuilder(wf.basis, aux), df=True))["corr"]
        e_ex = mp2_energy(sp, MOIntegrals(sp, _core.FockBuilder(wf.basis, 1e-12)))["corr"]
        assert e_df == pytest.approx(e_ex, abs=2e-4)
        assert e_df != pytest.approx(e_ex, abs=1e-7)


def _run(tmp_path, tasks, system="w: {coordinates: h2o.xyz, basis: def2-SVP}"):
    for f in ("h2o.xyz", "ch2.xyz"):
        (tmp_path / f).write_text((DATA / f).read_text())
    inp = tmp_path / "job.yaml"
    inp.write_text(f"systems:\n  {system}\ntasks:\n" + tasks)
    return [r for _, r in gemqc.run(inp, quiet=True)]


def test_task(tmp_path):
    """Task results: components, frozen core per irrep, pair energies on HDF5; symmetry
    against C1; df-mp2 with the correlation fitting basis; frozen off lowers the energy."""
    res = _run(tmp_path, "  - {system: w, task: scf, convergence: 1e-9}\n"
                         "  - {task: mp2, pair_energies: true}\n"
                         "  - {task: df-mp2}\n  - {task: mp2, frozen: off}\n")
    scf_, mp2, dfmp2, allel = res
    assert mp2["method"] == "MP2" and mp2["reference"] == "RHF"
    assert mp2["frozen_per_irrep"] == [1, 0, 0, 0]
    assert mp2["occupied_per_irrep"] == {"inner_core": [0, 0, 0, 0], "outer_core": [0, 0, 0, 0],
                                         "valence": [2, 1, 1, 0]}
    assert allel["occupied_per_irrep"]["outer_core"] == [1, 0, 0, 0]  # O 1s
    # core correlation (O 1s correlated): parts add up; frozen core: no parts
    assert sum(allel[k] for k in ("e_core", "e_core_valence", "e_valence")) == pytest.approx(
        allel["correlation_energy"], abs=1e-12)
    assert allel["e_core"] < 0 and allel["e_core_valence"] < 0 and "e_core" not in mp2
    assert mp2["correlation_energy"] == pytest.approx(mp2["e_os"] + mp2["e_ss"], abs=1e-14)
    assert mp2["energy"] == pytest.approx(scf_["energy"] + mp2["correlation_energy"], abs=1e-12)
    assert mp2["correlation_energy"] == pytest.approx(-0.201336262998, abs=1e-9)  # regression
    assert dfmp2["density_fitting"]["c_auxbasis"] == "def2-SVP-C"
    assert dfmp2["correlation_energy"] == pytest.approx(mp2["correlation_energy"], abs=2e-4)
    assert allel["correlation_energy"] < mp2["correlation_energy"] - 1e-3
    json.loads((tmp_path / "job.json").read_text())
    c1 = _run(tmp_path, "  - {system: w, task: scf, convergence: 1e-9}\n  - {task: mp2}\n",
              "w: {coordinates: h2o.xyz, basis: def2-SVP, symmetry: nosym}")
    assert c1[1]["correlation_energy"] == pytest.approx(mp2["correlation_energy"], abs=1e-10)
    assert c1[1]["frozen_per_irrep"] == [1]
    res = _run(tmp_path, "  - {system: w, task: scf}\n  - {task: mp2, pair_energies: true}\n")
    with h5py.File(tmp_path / "job.h5") as f:
        e = f["w/mp2/pair_energies_singlet"][()] + f["w/mp2/pair_energies_triplet"][()]
        assert e.sum() == pytest.approx(res[1]["correlation_energy"], abs=1e-12)
    p = c1[1]["pair_energies"]  # in the results also without pair_energies: true
    assert len(p["i"]) == len(p["singlet"]) == len(p["triplet"]) == 10  # 4 active occupied
    assert sum(p["singlet"]) + sum(p["triplet"]) == pytest.approx(
        c1[1]["correlation_energy"], abs=1e-12)
    assert all(t == 0.0 for i, j, t in zip(p["i"], p["j"], p["triplet"]) if i == j)


def test_open_shell_task(tmp_path):
    """ROHF (semicanonical, singles) and UHF references of triplet CH2."""
    sysline = "c: {coordinates: ch2.xyz, basis: def2-SVP, multiplicity: 3}"
    rohf = _run(tmp_path, "  - {system: c, task: rohf}\n  - {task: mp2}\n"
                          "  - {task: mp2, semicanonical: diag_first}\n"
                          "  - {task: mp2, frozen: off}\n", sysline)
    uhf = _run(tmp_path, "  - {system: c, task: uhf}\n  - {task: mp2}\n"
                         "  - {task: mp2, frozen: off}\n", sysline)
    parts = ("e_core", "e_core_valence", "e_valence", "e_singles")
    for r in (rohf[3], uhf[2]):  # core correlation: parts (and singles) add up
        assert sum(r[k] for k in parts) == pytest.approx(r["correlation_energy"], abs=1e-12)
    assert rohf[1]["reference"] == "ROHF" and uhf[1]["reference"] == "UHF"
    assert rohf[1]["e_singles"] < 0 and uhf[1]["e_singles"] == 0.0
    assert rohf[1]["correlation_energy"] != pytest.approx(rohf[2]["correlation_energy"], abs=1e-8)
    assert rohf[2]["correlation_energy"] == pytest.approx(rohf[1]["correlation_energy"], abs=1e-3)


def test_errors(tmp_path):
    with pytest.raises(InputError, match="needs an SCF task"):
        _run(tmp_path, "  - {system: w, task: mp2}\n")
    with pytest.raises(InputError, match="no correlation fitting basis for 'cc-pVDZ'"):
        _run(tmp_path, "  - {system: w, task: scf}\n  - {task: df-mp2}\n",
             "w: {coordinates: h2o.xyz, basis: cc-pVDZ}")
    with pytest.raises(InputError, match="frozen: 9 orbitals requested"):
        _run(tmp_path, "  - {system: w, task: scf}\n  - {task: mp2, frozen: 9}\n")


# --- gradients -------------------------------------------------------------------------

def _fd_setup(R, df, frozen):
    from gemqc.correlation import orbital_spaces as spaces_of
    mol0 = gemqc.read_xyz(DATA / "h2o.xyz")
    mol = gemqc.Molecule()
    for z, r in zip(mol0.atomic_numbers, R):
        mol.add_atom(int(z), list(r))
    bs = gemqc.build_basis(mol, "def2-SVP")
    jk = _core.DFBuilder(bs, gemqc.build_basis(mol, "def2-universal-JK")) if df else None
    ri = _core.DFBuilder(bs, gemqc.build_basis(mol, "def2-SVP-C")) if df else None
    o = gemqc.ScfOptions()
    o.convergence, o.energy_convergence = 1e-10, 1e-12
    wf = gemqc.rhf(mol, bs, o, gemqc.sad_guess(mol, bs), jk=jk)
    return wf, spaces_of(wf, frozen=frozen), jk, ri


def test_open_shell_references(tmp_path):
    """Triplet CH2 cc-pVTZ: DF-RMP2 (ROHF, semicanonical, singles; spin Fock matrices with
    the JK fitting of the DF-SCF) against Molpro 2025.5, DF-UMP2 against Turbomole."""
    (tmp_path / "ch2.xyz").write_text((DATA / "ch2.xyz").read_text())
    inp = tmp_path / "c.yaml"
    inp.write_text("systems:\n  c: {coordinates: ch2.xyz, basis: cc-pVTZ, multiplicity: 3}\n"
                   "tasks:\n  - {system: c, task: df-rohf, convergence: 1e-9}\n"
                   "  - {task: df-mp2}\n  - {task: df-uhf, convergence: 1e-9}\n"
                   "  - {task: df-mp2}\n")
    res = [r for _, r in gemqc.run(inp, quiet=True)]
    rohf, rmp2, uhf, ump2 = res
    assert rohf["energy"] == pytest.approx(-38.930772567751, abs=1e-9)
    assert rmp2["correlation_energy"] == pytest.approx(-0.123566924340, abs=1e-9)
    assert rmp2["e_singles"] == pytest.approx(-0.003006416956, abs=1e-9)
    assert rmp2["e_os"] == pytest.approx(-0.094201577, abs=1e-9)
    assert rmp2["e_ss"] == pytest.approx(-0.024072589 - 0.002286341, abs=2e-9)
    # Turbomole (jkbas differs: SCF 1.3e-7 apart)
    assert ump2["energy"] == pytest.approx(-39.0544493743, abs=3e-7)
    assert ump2["energy"] - uhf["energy"] == pytest.approx(-39.0544493743 + 38.93645250252,
                                                           abs=2e-8)


