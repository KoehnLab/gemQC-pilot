"""Density fitting: exact limit, hybrid storage, fitting error, df- tasks, auxiliary basis table."""

import json
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.basislib import build_basis, jk_auxbasis

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def quiet():
    _core.set_output_enabled(False)
    yield
    _core.set_output_enabled(True)
    _core.set_memory(0)


def he_exact_aux():
    """He with s functions; the fitting basis contains all their products."""
    mol = gemqc.Molecule("he")
    mol.add_atom(2, [0.1, 0.2, 0.3])
    ex = [8.0, 2.0, 0.5]
    bs = _core.BasisSet(mol, {2: [_core.ShellDef(0, True, [a], [1.0]) for a in ex]})
    aux = _core.BasisSet(mol, {2: [_core.ShellDef(0, True, [a + b], [1.0])
                                   for i, a in enumerate(ex) for b in ex[i:]]})
    return bs, aux


def test_exact_limit():
    bs, aux = he_exact_aux()
    fb, df = _core.FockBuilder(bs, 1e-14), _core.DFBuilder(bs, aux, 1e-14)
    D = np.random.default_rng(0).standard_normal((3, 3))
    for dens, sym in (([D + D.T], True), ([D], False)):
        (J1,), (K1,) = fb.compute(dens, sym)
        (J2,), (K2,) = df.compute(dens, sym)
        assert np.abs(J1 - J2).max() < 1e-12 and np.abs(K1 - K2).max() < 1e-12
    C = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 2)))[0]
    for a, b in zip(sum(fb.pair_jk(C), []), sum(df.pair_jk(C), [])):
        assert np.abs(a - b).max() < 1e-12


def test_hybrid_storage_and_fitting_error():
    """All stored, partly stored and all direct give the same J/K; errors vs exact small."""
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    bs, aux = build_basis(mol, "def2-SVP"), build_basis(mol, "def2-universal-JK")
    D = np.asarray(gemqc.rhf(mol, bs).D)
    Dn = np.random.default_rng(2).standard_normal(D.shape) * 0.1  # non-symmetric
    full = _core.DFBuilder(bs, aux)
    assert full.blocks_direct == 0 and full.blocks_stored > 0
    _core.set_memory(full.stored_bytes)  # budget: half of it (memory_fraction 0.5)
    part = _core.DFBuilder(bs, aux)
    _core.set_memory(0)
    direct = _core.DFBuilder(bs, aux, memory_fraction=0.0)
    assert 0 < part.blocks_direct < full.blocks_stored and direct.blocks_stored == 0
    ref = full.compute([D, Dn], False)
    for b in (part, direct):
        for x, y in zip(sum(ref, []), sum(b.compute([D, Dn], False), [])):
            assert np.abs(x - y).max() < 1e-12
    (Je,), (Ke,) = _core.FockBuilder(bs, 1e-12).compute([D])
    (J,), (K,) = full.compute([D])
    assert abs(0.5 * np.sum(D * (J - Je))) < 1e-4 and abs(0.25 * np.sum(D * (K - Ke))) < 1e-4


def test_fitted_storage():
    """Storage fitted (B = M^T (P|mu nu)) gives the raw results; auto follows the budget."""
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    bs, aux = build_basis(mol, "def2-SVP"), build_basis(mol, "def2-universal-JK")
    wf = gemqc.rhf(mol, bs)
    D = np.asarray(wf.D)
    Dn = np.random.default_rng(2).standard_normal(D.shape) * 0.1  # non-symmetric
    raw, fit = _core.DFBuilder(bs, aux), _core.DFBuilder(bs, aux, storage="fitted")
    assert fit.fitted and not raw.fitted and fit.blocks_stored == 0
    for dens, sym in (([D], True), ([D, Dn], False)):
        for x, y in zip(sum(raw.compute(dens, sym), []), sum(fit.compute(dens, sym), [])):
            assert np.abs(x - y).max() < 1e-11
    C = np.asarray(wf.C)
    for x, y in zip(sum(raw.pair_jk(C[:, 2:5]), []), sum(fit.pair_jk(C[:, 2:5]), [])):
        assert np.abs(x - y).max() < 1e-11
    B = raw.mo_three_index(C, C[:, 2:5])
    assert np.abs(B - fit.mo_three_index(C, C[:, 2:5])).max() < 1e-11
    for x, y in zip(raw.fitted_coulomb(B[:, :3]), fit.fitted_coulomb(B[:, :3])):
        assert np.abs(x - y).max() < 1e-11
    assert _core.DFBuilder(bs, aux, storage="auto").fitted  # no limit
    _core.set_memory(raw.stored_bytes)  # budget: half of B
    assert not _core.DFBuilder(bs, aux, storage="auto").fitted
    with pytest.raises(ValueError):
        _core.DFBuilder(bs, aux, storage="packed")


def run(tmp_path, text, name="w", **kw):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    (tmp_path / "ch2.xyz").write_text((DATA / "ch2.xyz").read_text())
    inp = tmp_path / f"{name}.yaml"
    inp.write_text(text)
    return [r for _, r in gemqc.run(inp, quiet=True, **kw)]


def test_df_scf_tasks(tmp_path):
    res = run(tmp_path, "systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
              "  n: {coordinates: h2o.xyz, basis: def2-SVP, symmetry: nosym}\n"
              "  c: {coordinates: ch2.xyz, basis: def2-SVP, multiplicity: 3, modify: adapt, "
              "jk_auxbasis: def2-universal-JK}\n"
              "tasks:\n  - {system: w, task: df-scf}\n  - {task: scf}\n"
              "  - {system: n, task: df-rhf}\n"
              "  - {system: c, task: df-rohf}\n  - {task: rohf}\n  - {task: df-uhf}\n"
              "")
    df, ex, nosym, rohf_df, rohf, uhf_df = res
    assert df["method"] == "DF-RHF" and df["density_fitting"]["jk_auxbasis"] == "def2-universal-JK"
    assert abs(df["energy"] - ex["energy"]) < 1e-4 and ex["iterations"] <= 4  # restart
    assert nosym["energy"] == pytest.approx(df["energy"], abs=1e-8)
    assert abs(rohf_df["energy"] - rohf["energy"]) < 1e-4
    assert uhf_df["converged"] and uhf_df["energy"] < rohf_df["energy"]
    js = json.loads((tmp_path / "w.json").read_text())["tasks"][0]
    assert js["task"] == "df-scf" and js["results"]["density_fitting"]["naux"] == 113
    assert js["results"]["density_fitting"]["jk_storage"] == "fitted"  # auto, no limit
    raw, fit, again = run(tmp_path, "systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                          "tasks:\n  - {system: w, task: df-scf, jk_storage: raw}\n"
                          "  - {task: df-scf, jk_storage: fitted}\n  - {task: df-scf}\n", "f")
    assert [r["density_fitting"]["jk_storage"] for r in (raw, fit, again)] == ["raw", "fitted", "fitted"]
    assert raw["energy"] == pytest.approx(df["energy"], abs=1e-9)
    assert fit["energy"] == pytest.approx(df["energy"], abs=1e-9)
    with pytest.raises(gemqc.InputError):
        run(tmp_path, "systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
            "tasks:\n  - {system: w, task: scf, jk_storage: raw}\n", "e")


def test_aux_basis_and_errors(tmp_path):
    assert jk_auxbasis("def2-SVP") == ("def2-universal-JK", "default for def2-SVP (auxbasis.yaml)")
    assert jk_auxbasis("cc-pVDZ")[0] == "cc-pVTZ-JK"                 # DZ: TZ set
    assert jk_auxbasis("cc-pVTZ") == ("cc-pVTZ-JK", "file <basis>-JK")  # file in lib/basis
    assert jk_auxbasis("def2-SVP", "x-JK") == ("x-JK", "input (jk_auxbasis)")
    with pytest.raises(ValueError, match="no JK fitting basis"):
        jk_auxbasis("ANO-R0")
    head = "systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\ntasks:\n"
    with pytest.raises(gemqc.InputError, match="no density-fitted variant"):
        run(tmp_path, head + "  - {system: w, task: df-oneint}\n")
    with pytest.raises(gemqc.InputError, match="density fitting"):
        run(tmp_path, head.replace("def2-SVP}", "def2-SVP, jk_auxbasis: none-JK}")
            + "  - {system: w, task: df-scf}\n")


def test_butane_turbomole_reference(tmp_path):
    """DF-RHF/def2-SVP with def2-universal-JK vs Turbomole ridft ($rik, jkbas def2-SVP)."""
    (tmp_path / "butane.xyz").write_text((DATA / "butane.xyz").read_text())
    (res,) = run(tmp_path, "systems:\n  b: {coordinates: butane.xyz, basis: def2-SVP}\n"
                 "tasks:\n  - {system: b, task: df-rhf, convergence: 1e-8}\n", "b")
    assert res["energy"] == pytest.approx(-157.1843555094, abs=1e-9)
