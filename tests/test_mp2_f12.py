"""DF-MP2-F12 with fixed amplitudes (task df-mp2-f12): Molpro references for 3C, 3*C,
NOZ, HY1 (water aug-cc-pVTZ, reference_f12/h2o-dfmp2-f12-molpro and -testcabs-molpro),
Turbomole fixed-A for 3A, symmetry, storage, pair energies, geminal options."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gemqc
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"
WATER = ("w:\n    basis: {basis}\n    symmetry: {sym}\n    geometry: |\n"
         "      O     0.0000000    0.0000000    0.3935881\n"
         "      H     0.7571222    0.0000000   -0.1967941\n"
         "      H    -0.7571222    0.0000000   -0.1967941")


def _run(tmp_path, tasks, basis="aug-cc-pVTZ", sym="nosym"):
    inp = tmp_path / "job.yaml"
    inp.write_text("systems:\n  " + WATER.format(basis=basis, sym=sym) + "\ntasks:\n" + tasks)
    return [r for _, r in gemqc.run(inp, quiet=True)]


def test_molpro_water(tmp_path):
    """Water aug-cc-pVTZ, STG-6G (gamma = 1), RI basis AVTZ/JKFIT, MP2FIT fitting, frozen
    core excluded from the CABS singles: Molpro DF-MP2-F12 3C(FIX), 3*C(FIX), NOZ,
    NOZ+HY1 (singlet, triplet parts); all electrons 3C(FIX,NOZ); with all-electron CABS
    singles Turbomole's fixed-A (ricc2, same geminal fit) for 3A."""
    res = _run(tmp_path,
               "  - {system: w, task: df-rhf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12, cabs_auxbasis: aug-cc-pVTZ-JK, all_electron_cabs: false}\n"
               "  - {task: df-mp2-f12, all_electron_cabs: false, noz: true}\n"
               "  - {task: df-mp2-f12, all_electron_cabs: false, noz: true, hy1: true}\n"
               "  - {task: df-mp2-f12, frozen: off, noz: true}\n"
               "  - {task: df-mp2-f12, approximation: 3a}\n")
    _, full, noz, hy1, ae, a = res
    tol = 1e-10
    assert full["energy"] == pytest.approx(-76.362038370049, abs=tol)
    assert full["e_f12_3c"] == pytest.approx(-0.028351809431, abs=tol)
    assert full["e_f12_3sc"] == pytest.approx(-0.028237504396, abs=tol)
    assert full["e_f12_singlet"] == pytest.approx(-0.024563723456, abs=tol)
    assert full["e_f12_triplet"] == pytest.approx(-0.003788085975, abs=tol)
    assert full["e_mp2_singlet"] == pytest.approx(-0.170744459746, abs=tol)
    assert full["e_mp2_triplet"] == pytest.approx(-0.097816176356, abs=tol)
    assert full["approximation"] == "3C(FIX)"
    assert noz["e_f12_3c"] == pytest.approx(-0.028367631242, abs=tol)
    assert noz["e_f12_3sc"] == pytest.approx(-0.028253326206, abs=tol)
    assert hy1["e_f12_3c"] == pytest.approx(-0.027859057414, abs=tol)
    assert hy1["e_f12_3sc"] == pytest.approx(-0.027744752378, abs=tol)
    assert hy1["approximation"] == "3C(FIX,NOZ,HY1)"
    assert ae["e_cabs_singles"] == pytest.approx(-0.004881362855, abs=tol)
    assert ae["e_f12_3c"] == pytest.approx(-0.040280496935, abs=tol)
    assert ae["e_f12_3sc"] == pytest.approx(-0.041028280861, abs=tol)
    assert ae["energy"] == pytest.approx(-76.389250236965, abs=tol)
    assert a["e_f12"] == pytest.approx(-0.031808655311, abs=5e-10)
    assert a["e_f12_singlet"] == pytest.approx(-0.027535809766, abs=5e-10)
    assert a["e_f12_triplet"] == pytest.approx(-0.004272845544, abs=5e-10)


def test_symmetry_storage_pairs(tmp_path):
    """C2v equals C1; disk storage equals memory; pair energies sum to the totals; 3*c
    selects the uncoupled energy; explicit geminal exponents (Molpro's printed fit) and
    exact Slater integrals give close F12 corrections."""
    tasks = ("  - {system: w, task: df-rhf, convergence: 1e-10}\n"
             "  - {task: df-mp2-f12, pair_energies: true}\n"
             "  - {task: df-mp2-f12, integrals: disk, approximation: 3*c}\n"
             "  - {task: df-mp2-f12, geminal_exponents: [0.19532, 0.81920, 2.85917, 9.50073,"
             " 35.69989, 197.79328], geminal_coefficients: [0.27070, 0.30552, 0.18297,"
             " 0.10986, 0.06810, 0.04224]}\n"
             "  - {task: df-mp2-f12, geminal: exact}\n")
    c1 = _run(tmp_path, tasks, basis="aug-cc-pVDZ")
    c2v = _run(tmp_path, tasks, basis="aug-cc-pVDZ", sym="auto")
    for a, b in zip(c1[1:], c2v[1:]):
        assert a["energy"] == pytest.approx(b["energy"], abs=1e-10)
    _, f12, disk, expl, exact = c1
    assert disk["e_f12_3c"] == pytest.approx(f12["e_f12_3c"], abs=1e-13)
    assert disk["e_f12"] == pytest.approx(disk["e_f12_3sc"], abs=1e-15)
    assert disk["approximation"] == "3*C(FIX)"
    assert expl["e_f12"] == pytest.approx(f12["e_f12"], abs=1e-6)
    assert exact["e_f12"] == pytest.approx(f12["e_f12"], abs=2e-3)
    assert exact["e_f12"] != pytest.approx(f12["e_f12"], abs=1e-6)
    assert f12["e_f12"] < 0.0


def test_pair_energy_sums(tmp_path):
    """Singlet and triplet pair energies (HDF5, JSON by default) sum to the MP2 and F12
    energies; the MP2 pairs equal those of df-mp2; core, core-valence and valence parts
    (all electrons) add up."""
    res = _run(tmp_path, "  - {system: w, task: df-rhf, convergence: 1e-10}\n"
                         "  - {task: df-mp2-f12, frozen: off}\n"
                         "  - {task: df-mp2-f12, pair_energies: true}\n  - {task: df-mp2}\n"
                         "  - {task: df-mp2-f12, approximation: 3*c}\n", basis="aug-cc-pVDZ")
    _, ae, f12, mp2, f3s = res  # the HDF5 group holds the last task (3*c)
    for x in ("mp2", "f12"):  # core correlation: MP2 and F12 parts add up
        parts = [ae[f"e_{x}_{p}"] for p in ("core", "core_valence", "valence")]
        total = ae["mp2_correlation_energy"] if x == "mp2" else ae["e_f12"]
        assert sum(parts) == pytest.approx(total, abs=1e-12)
    assert "e_mp2_core" not in f12
    with h5py.File(tmp_path / "job.h5") as h:
        g = h["w/mp2_f12"]
        pm, p3s = np.asarray(g["pair_energies_mp2"]), np.asarray(g["pair_energies_3sc"])
    assert pm.sum() == pytest.approx(f12["mp2_correlation_energy"], abs=1e-12)
    assert p3s.sum() == pytest.approx(f3s["e_f12"], abs=1e-12)
    assert pm[..., 0].sum() == pytest.approx(f12["e_mp2_singlet"], abs=1e-12)
    for r in (f12, f3s):
        p = r["pair_energies"]
        for part in ("singlet", "triplet"):
            assert sum(p[f"mp2_{part}"]) == pytest.approx(r[f"e_mp2_{part}"], abs=1e-12)
            assert sum(p[f"f12_{part}"]) == pytest.approx(r[f"e_f12_{part}"], abs=1e-12)
    q = mp2["pair_energies"]
    assert np.abs(np.subtract(q["singlet"], f12["pair_energies"]["mp2_singlet"])).max() < 1e-12
    assert np.abs(np.subtract(q["triplet"], f12["pair_energies"]["mp2_triplet"])).max() < 1e-12


def _he2():
    """He2 (3 bohr) cc-pVDZ, RHF, its RI space with aug-cc-pVDZ-CABS (exact Fock)."""
    from gemqc import _core
    from gemqc.cabs import build_ri_space
    from gemqc.correlation import orbital_spaces
    mol = _core.Molecule()
    for z in ([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]):
        mol.add_atom(2, z)
    bs = gemqc.build_basis(mol, "cc-pVDZ")
    o = gemqc.ScfOptions()
    o.convergence = 1e-10
    wf = gemqc.rhf(mol, bs, o, gemqc.sad_guess(mol, bs))
    a = orbital_spaces(wf, frozen="off", deleted="off").alpha
    return mol, a, build_ri_space(mol, "cc-pVDZ", a.C, a.nocc, "aug-cc-pVDZ-CABS")


def test_exact_pair_integrals(quiet):
    """Exact pair integrals <ij|g|p'q'> (FockBuilder with the F12 operators, pair_jk)
    against the full AO tensor of the RI basis; memory and disk storage."""
    from gemqc import _core
    from gemqc.f12 import ExactF12Integrals
    from gemqc.geminal import Geminal
    mol, a, ri = _he2()
    gem = Geminal.make(1.0, 6)
    ops = gem.operators()
    C = ri.C
    full = {}
    for name in ("coulomb", "f12", "f12^2", "f12/r12", "(grad f12)^2"):
        V = _core.two_body_4c(ops[name], ri.basis)
        full[name] = np.einsum("mnls,mi,nP,lj,sQ->ijPQ", V, C[:, :2], C, C[:, :2], C,
                               optimize=True)  # (iP|jQ) = <ij|PQ>
    for mode in ("memory", "disk"):
        ex = ExactF12Integrals(ri, 0, 2, gem, 1e-14, mode)
        for i, j in ((0, 0), (1, 0), (1, 1)):
            p = ex.pair(i, j)
            assert np.abs(p["F"] - full["f12"][i, j]).max() < 1e-12
            G, Gref, nmo = p["G"], full["coulomb"][i, j], ri.nmo  # (v x), (x y) not needed
            assert np.abs(G[:, :nmo] - Gref[:, :nmo]).max() < 1e-12
            assert np.abs(G[:2, nmo:] - Gref[:2, nmo:]).max() < 1e-12
            assert np.abs(p["F2c"]["ij"] - full["f12^2"][i, j][:, :2]).max() < 1e-12
            assert np.abs(p["F2c"]["ji"] - full["f12^2"][j, i][:, :2]).max() < 1e-12
            for name in ("f12/r12", "(grad f12)^2"):
                ref = full[name][i, j][i, j] if i == j else full[name][i, j][[i, j]][:, [j, i]]
                blk = p[name]
                assert abs(blk[0, 0] - full[name][i, j][i, j]) < 1e-12
                if i != j:
                    assert abs(blk[0, 1] - full[name][i, j][j, i]) < 1e-12
                    assert abs(blk[1, 0] - full[name][j, i][i, j]) < 1e-12
                    assert abs(blk[1, 1] - full[name][j, i][j, i]) < 1e-12
        ex.close()


def test_exact_task(tmp_path):
    """mp2-f12 (exact) against df-mp2-f12: F12 corrections within the DF error; open
    shells refused."""
    res = _run(tmp_path, "  - {system: w, task: scf, convergence: 1e-10}\n"
                         "  - {task: mp2-f12}\n  - {task: df-mp2-f12}\n",
               basis="aug-cc-pVDZ", sym="auto")
    _, ex, df = res
    assert ex["method"] == "MP2-F12" and "density_fitting" not in ex
    # DF error 5e-5, mostly from the fitted Fock matrices in the RI basis (aug-cc-pVTZ-JK;
    # 1.4e-5 with aug-cc-pVQZ-JK and aug-cc-pVQZ-C)
    assert ex["e_f12"] == pytest.approx(df["e_f12"], abs=1e-4)
    assert ex["e_f12_3sc"] == pytest.approx(df["e_f12_3sc"], abs=1e-4)
    assert ex["mp2_correlation_energy"] == pytest.approx(df["mp2_correlation_energy"], abs=1e-4)
    assert ex["e_cabs_singles"] == pytest.approx(df["e_cabs_singles"], abs=1e-5)
    inp = tmp_path / "open.yaml"
    inp.write_text("systems:\n  c: {geometry: 'C 0 0 0', basis: aug-cc-pVDZ, multiplicity: 3}\n"
                   "tasks:\n  - {system: c, task: rohf}\n  - {task: df-mp2-f12}\n")
    with pytest.raises(gemqc.InputError):
        list(gemqc.run(inp, quiet=True))


def test_approximation_a(tmp_path):
    """3A and 3*A (water aug-cc-pVDZ): a value computes its pair only, a list all named
    pairs with the first as the main result; DF and exact regression values, exact against
    DF within the DF error, the F C term and the coupling as in 3C; NOZ without
    approximation C and invalid values refused."""
    res = _run(tmp_path, "  - {system: w, task: df-rhf, convergence: 1e-10}\n"
                         "  - {task: df-mp2-f12, cabs_auxbasis: aug-cc-pVTZ-JK}\n"
                         "  - {task: df-mp2-f12, approximation: 3a}\n"
                         "  - {task: df-mp2-f12, approximation: [3*a, 3c], noz: true}\n"
                         "  - {task: mp2-f12, approximation: 3*a}\n"
                         "  - {task: mp2-f12, approximation: [3c, 3a]}\n",
               basis="aug-cc-pVDZ")
    _, c, df, mix, ex, exmix = res
    assert "e_f12_3a" not in c and "e_f12_3c" not in df and "e_f12_3sc" not in ex
    assert df["approximation"] == "3A(FIX)" and ex["approximation"] == "3*A(FIX)"
    assert mix["approximation"] == "3*A(FIX)" and exmix["approximation"] == "3C(FIX)"
    assert df["e_f12"] == pytest.approx(df["e_f12_3a"], abs=1e-15)
    assert ex["e_f12"] == pytest.approx(ex["e_f12_3sa"], abs=1e-15)
    assert mix["e_f12"] == pytest.approx(mix["e_f12_3sa"], abs=1e-15)
    assert exmix["e_f12"] == pytest.approx(exmix["e_f12_3c"], abs=1e-15)
    for k in ("e_f12_3a", "e_f12_3sa"):
        assert mix[k] == pytest.approx(df[k], abs=1e-13)
        assert exmix[k] == pytest.approx(ex[k], abs=1e-13)
    assert df["e_f12_3a"] == pytest.approx(-0.085232217443, abs=1e-9)
    assert df["e_f12_3sa"] == pytest.approx(-0.086680351159, abs=1e-9)
    assert ex["e_f12_3a"] == pytest.approx(-0.085184161408, abs=1e-9)
    assert ex["e_f12_3sa"] == pytest.approx(-0.086632117238, abs=1e-9)
    for r in (mix, exmix):
        assert r["e_f12_3a"] - r["e_f12_3sa"] == pytest.approx(
            r["e_f12_3c"] - r["e_f12_3sc"], abs=1e-12)
    assert exmix["e_f12_3c"] == pytest.approx(c["e_f12_3c"], abs=1e-4)
    assert ex["e_f12_3a"] == pytest.approx(df["e_f12_3a"], abs=1e-4)
    for task in ("{task: df-mp2-f12, approximation: 3a, noz: true}",
                 "{task: df-mp2-f12, approximation: [3c, 3c]}",
                 "{task: df-mp2-f12, approximation: 3b}",
                 "{task: mp2-f12xg-pilot, gammas: [1.0, 1.5], bf_groups: core_valence,"
                 " approximation: [3c, 3a]}"):
        inp = tmp_path / "bad.yaml"
        inp.write_text("systems:\n  " + WATER.format(basis="aug-cc-pVDZ", sym="nosym")
                       + "\ntasks:\n  - {system: w, task: df-rhf}\n  - " + task + "\n")
        with pytest.raises(gemqc.InputError):
            list(gemqc.run(inp, quiet=True))


def test_size_consistency_he_ne(tmp_path):
    """He ... Ne at 50 and 100 Angstrom against He + Ne: mp2-f12 is additive (CABS singles,
    MP2, F12); df-mp2-f12 only up to a deviation that decays as 1/R (the Coulomb-metric
    fit couples the fitting errors of the fragments through their charges; docs: mp2-f12,
    density fitting and size consistency)."""
    basis = "basis: aug-cc-pVDZ, symmetry: nosym, jk_auxbasis: def2-universal-JK"
    inp = tmp_path / "hene.yaml"
    inp.write_text(
        "systems:\n"
        f"  d50: {{{basis}, geometry: \"He 0 0 0\\nNe 0 0 50\"}}\n"
        f"  d100: {{{basis}, geometry: \"He 0 0 0\\nNe 0 0 100\"}}\n"
        f"  he: {{{basis}, geometry: 'He 0 0 0'}}\n"
        f"  ne: {{{basis}, geometry: 'Ne 0 0 0'}}\n"
        "tasks:\n" + "".join(
            f"  - {{system: {s}, task: scf, convergence: 1e-10}}\n"
            "  - {task: df-mp2-f12}\n  - {task: mp2-f12}\n" for s in ("d50", "d100", "he", "ne")))
    res = [r for _, r in gemqc.run(inp, quiet=True)]
    (_, df50, ex50), (_, df100, _), (_, dfhe, exhe), (_, dfne, exne) = (
        res[i:i + 3] for i in range(0, 12, 3))
    keys = ("e_cabs_singles", "mp2_correlation_energy", "e_f12", "e_f12_3sc", "energy")
    for k in keys:
        assert ex50[k] == pytest.approx(exhe[k] + exne[k], abs=1e-11), k
    for k in ("e_cabs_singles", "e_f12"):
        d50, d100 = (d[k] - dfhe[k] - dfne[k] for d in (df50, df100))
        assert abs(d50) < 5e-7, k            # 3.6e-8 (CABS singles), 1.9e-7 (F12)
        assert 1.8 < d50 / d100 < 2.2, k     # 1/R
    assert df50["mp2_correlation_energy"] == pytest.approx(
        dfhe["mp2_correlation_energy"] + dfne["mp2_correlation_energy"], abs=1e-9)
