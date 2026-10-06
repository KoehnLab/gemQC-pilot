"""F12XG (extended geminals): group geminals and their operators."""

import numpy as np
import pytest

from gemqc import _core
from gemqc.geminal import Geminal, XGGeminal
from test_qn import quiet  # noqa: F401 (fixture)

R = np.linspace(0.05, 6.0, 120)


def _value(op, r):
    """scale * g(r) of a cgtg or stg operator."""
    if op.kind == "stg":
        return op.scale * np.exp(-op.zeta * r)
    assert op.kind == "cgtg"
    return op.scale * sum(c * np.exp(-a * r * r) for a, c in op.gtg)


def _same(o1, o2):
    return ((o1.kind, o1.zeta, o1.scale, o1.gtg, o1.gtg2) ==
            (o2.kind, o2.zeta, o2.scale, o2.gtg, o2.gtg2))


needs_kernel = pytest.mark.skipif(not _core.TwoBodyOperator.has_delcgtgab(),
                                  reason="libint without delcgtga_delcgtgb")


@pytest.mark.parametrize("fit", [6, "exact"])
def test_equal_gammas_reduce_to_standard(fit):
    """Equal group exponents: every group pair has the standard geminal and operators
    (identical, not only close), products are f12^2."""
    std = Geminal.make(1.1, fit).operators()
    xg = XGGeminal([1.1, 1.1], fit)
    assert xg.pairs == [(0, 0), (0, 1), (1, 1)]
    for G, H in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for name, op in xg.operators(G, H).items():
            assert _same(op, std[name])
        for bra in xg.pairs:
            prod = xg.product_operators(bra, (G, H))
            assert _same(prod["f12 x f12"], std["f12^2"])
            assert _same(prod["grad f12 . grad f12"], std["(grad f12)^2"])


@pytest.mark.parametrize("fit", [6, "exact"])
def test_group_pair_geminals(fit):
    """Distinct exponents: gamma_GH = (gt_G + gt_H)/2, symmetric in G, H; the products
    equal f_G'H'(r) f_GH(r)."""
    gt = (0.8, 1.6)
    xg = XGGeminal(gt, fit)
    for G in range(2):
        for H in range(2):
            assert xg.geminal(G, H).gamma == pytest.approx(0.5 * (gt[G] + gt[H]), rel=1e-15)
            assert xg.geminal(G, H) is xg.geminal(H, G)
    f = {p: _value(xg.operators(*p)["f12"], R) for p in xg.pairs}
    for bra in xg.pairs:
        for ket in xg.pairs:
            prod = _value(xg.product_operators(bra, ket)["f12 x f12"], R)
            assert np.allclose(prod, f[bra] * f[ket], rtol=1e-12, atol=1e-14)
    # fitted geminals approximate -exp(-gamma r)/gamma (fit region)
    near = R <= 3.0
    for G, H in xg.pairs:
        g = xg.geminal(G, H).gamma
        assert np.abs(f[(G, H)] + np.exp(-g * R) / g)[near].max() < 5e-3 / g


# --- (grad g_a).(grad g_b) -------------------------------------------------------------

def _small_basis():
    """Mixed angular momenta on three centres (one primitive per shell)."""
    mol = _core.Molecule()
    for z, r in [(1, [0.0, 0.0, 0.0]), (2, [0.3, 1.2, -0.5]), (3, [-0.8, 0.1, 0.9])]:
        mol.add_atom(z, r)
    sh = {1: [(0, 1.2), (2, 0.7)], 2: [(1, 0.6), (3, 0.9)], 3: [(0, 0.3), (1, 2.1)]}
    lib = {z: [_core.ShellDef(l, True, [a], [1.0]) for l, a in s] for z, s in sh.items()}
    return _core.BasisSet(mol, lib)


@needs_kernel
def test_delcgtgab_kernel():
    """delcgtga_delcgtgb against the polarisation identity
    grad g_a . grad g_b = 1/2 [(grad(g_a + g_b))^2 - (grad g_a)^2 - (grad g_b)^2]
    (delcgtg2 only), its symmetry in a, b, and delcgtg2 for a = b."""
    from gemqc.geminal import stg_gtg_fit
    op = _core.TwoBodyOperator
    bs = _small_basis()
    ga = list(zip(*map(list, stg_gtg_fit(0.9, 6))))
    gb = list(zip(*map(list, stg_gtg_fit(1.7, 4))))

    def I(o):
        return _core.two_body_4c(o, bs)
    ab = I(op("delcgtga_delcgtgb", 0.0, 1.3, ga, gb))
    ref = 0.5 * 1.3 * (I(op("delcgtg2", 0.0, 1.0, ga + gb)) - I(op("delcgtg2", 0.0, 1.0, ga))
                       - I(op("delcgtg2", 0.0, 1.0, gb)))
    assert np.abs(ab).max() > 1e-2
    assert np.allclose(ab, ref, rtol=1e-10, atol=1e-13 * np.abs(ab).max())
    assert np.allclose(I(op("delcgtga_delcgtgb", 0.0, 1.3, gb, ga)), ab, rtol=1e-12,
                       atol=1e-15)
    assert np.allclose(I(op("delcgtga_delcgtgb", 0.0, 1.0, ga, ga)),
                       I(op("delcgtg2", 0.0, 1.0, ga)), rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        I(op("delcgtga_delcgtgb", 0.0, 1.0, ga, []))


@needs_kernel
def test_grad_product_fit_vs_exact():
    """The fitted (grad f_a).(grad f_b) of two group pairs approximates the exact Slater
    operator exp(-(g_a + g_b) r12)."""
    bs = _small_basis()
    fit, ex = XGGeminal([0.8, 1.6], 6), XGGeminal([0.8, 1.6], "exact")
    for bra, ket in [((0, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (0, 1))]:
        name = "grad f12 . grad f12"
        f = _core.two_body_4c(fit.product_operators(bra, ket)[name], bs)
        e = _core.two_body_4c(ex.product_operators(bra, ket)[name], bs)
        assert fit.product_operators(bra, ket)[name].kind == "delcgtga_delcgtgb"
        assert np.abs(f - e).max() < 1e-2 * np.abs(e).max()


# --- basis-function groups ---------------------------------------------------------------

def _mol(atoms):
    from gemqc import _core
    mol = _core.Molecule("m")
    for z, xyz in atoms:
        mol.add_atom(z, xyz)
    return mol


WATER = [(8, [0.0, 0.0, 0.2]), (1, [0.0, 1.43, -0.9]), (1, [0.0, -1.43, -0.9])]


def test_ao_r2_about_own_atom():
    """<r_A^2> does not depend on the position of the atom; the contracted 1s function of
    cc-pVQZ gives the core reference."""
    from gemqc.basislib import build_basis
    from gemqc.f12xg import CORE_R2, ao_r2
    r2 = {}
    for xyz in ([0.0, 0.0, 0.0], [1.3, -2.1, 0.7]):
        mol = _mol([(10, xyz)])
        r2[tuple(xyz)] = ao_r2(mol, build_basis(mol, "cc-pVQZ"))
    a, b = r2.values()
    assert np.allclose(a, b, rtol=1e-10)
    assert a.min() == pytest.approx(CORE_R2[10], abs=1e-4)


@pytest.mark.parametrize("basis,ncore", [("cc-pVTZ", 1), ("aug-cc-pVTZ", 1),
                                         ("def2-TZVPP", 1), ("def2-SVP", 1)])
def test_core_valence_water(basis, ncore):
    """core_valence (valence basis sets): only the O 1s function is core (def2-TZVPP: the
    innermost of the split 1s functions); H is valence."""
    from gemqc.basislib import build_basis
    from gemqc.f12xg import ao_shells, assign_bf_groups
    mol = _mol(WATER)
    bs = build_basis(mol, basis)
    g = assign_bf_groups(mol, bs, "core_valence")
    atoms = np.asarray(bs.bf_to_atom)
    l, _ = ao_shells(bs)
    core = g.mask(0)
    assert np.sum(core) == ncore
    assert np.all(atoms[core] == 0) and np.all(l[core] == 0)
    assert g.summary(mol, bs)[1].startswith(f"group 1: {ncore} functions (O1:{ncore})")


def test_bf_groups_atoms_functions():
    """Groups by atoms (numbers or element symbols) and by explicit labels; errors."""
    from gemqc.basislib import build_basis
    from gemqc.f12xg import assign_bf_groups
    mol = _mol(WATER)
    bs = build_basis(mol, "cc-pVDZ")
    atoms = np.asarray(bs.bf_to_atom)
    for spec in ({"atoms": [1]}, {"atoms": ["o"]}, {"atoms": ["1"]}):
        g = assign_bf_groups(mol, bs, spec)
        assert np.array_equal(g.labels, np.where(atoms == 0, 0, 1))
    g = assign_bf_groups(mol, bs, {"atoms": ["H"]})
    assert np.array_equal(g.labels, np.where(atoms == 0, 1, 0))
    lab = [1, 2] * (bs.nbf // 2) + [1] * (bs.nbf % 2)
    assert np.array_equal(assign_bf_groups(mol, bs, {"functions": lab}).labels,
                          np.array(lab) - 1)
    for bad in ("valence", {"atoms": [4]}, {"atoms": ["Ne"]}, {"functions": [1, 2]},
                {"orbitals": [1]}, {"atoms": 1}):
        with pytest.raises(ValueError):
            assign_bf_groups(mol, bs, bad)


def test_core_valence_cv_basis():
    """cc-pCVDZ-F12: the functions added to cc-pVDZ-F12 (s, p for Li-Ne; s, p, d for
    Na-Ar) are core, as is the contracted 1s (2s, 2p) function; the reference basis is
    derived from the name or given explicitly."""
    from gemqc.basislib import build_basis
    from gemqc.f12xg import assign_bf_groups, extra_functions
    mol = _mol([(3, [0.0, 0.0, 0.0]), (6, [0.0, 0.0, 6.0]), (11, [0.0, 6.0, 0.0]),
                (18, [6.0, 0.0, 0.0])])
    cv, v = build_basis(mol, "cc-pCVDZ-F12"), build_basis(mol, "cc-pVDZ-F12")
    extra = extra_functions(cv, v)
    atoms = np.asarray(cv.bf_to_atom)
    assert [np.sum(extra & (atoms == a)) for a in range(4)] == [4, 4, 9, 9]
    gv = assign_bf_groups(mol, v, "core_valence")
    ncore_v = [np.sum(gv.mask(0) & (np.asarray(v.bf_to_atom) == a)) for a in range(4)]
    # 1s; 1s, 2s, 2p (contracted, ratio <= 1); Na: also a compact p (ratio 0.46)
    assert ncore_v == [1, 1, 8, 5]
    for spec in ("core_valence", {"core_valence": "cc-pVDZ-F12"}):
        g = assign_bf_groups(mol, cv, spec)
        assert np.all(g.mask(0)[extra])
        assert [np.sum(g.mask(0) & (atoms == a)) for a in range(4)] == [5, 5, 17, 14]
    assert "26 functions not in cc-pVDZ-F12" in g.how
    with pytest.raises(ValueError):  # reference not contained
        assign_bf_groups(mol, v, {"core_valence": "cc-pCVDZ-F12"})


# --- exact F12XG pair integrals -------------------------------------------------------

LABELS = [1, 2, 1, 1, 2, 2, 1, 2, 2, 1]  # He2 cc-pVDZ: groups cut through the orbitals


def _he2_groups():
    from test_mp2_f12 import _he2
    from gemqc.f12xg import assign_bf_groups
    mol, a, ri = _he2()
    return a, ri, assign_bf_groups(mol, ri.obs_basis, {"functions": LABELS})


@pytest.mark.parametrize("fit", [6, "exact"])
def test_equal_gammas_pair_intermediates(fit, quiet):
    """gt_1 = gt_2: V, X, B, C of F12XG equal the standard ones for any groups."""
    from gemqc.f12 import ExactF12Integrals, PairIntermediates
    from gemqc.f12xg import ExactF12XGIntegrals
    a, ri, groups = _he2_groups()
    std = ExactF12Integrals(ri, 0, 2, Geminal.make(1.2, fit), 1e-14)
    xg = ExactF12XGIntegrals(ri, 0, 2, XGGeminal([1.2, 1.2], fit), groups, 1e-14)
    for noz, hy1 in ((False, False), (True, True)):
        ps, px = (PairIntermediates(x, ri, noz, hy1) for x in (std, xg))
        for i, j in ((0, 0), (1, 0), (1, 1)):
            ds, dx = ps.pair(i, j), px.pair(i, j)
            for key in ("V", "X", "B3sC", "B3C", "C", "K"):
                assert np.abs(ds[key] - dx[key]).max() < 1e-12, key
    std.close()
    xg.close()


def test_xg_pair_integrals_brute_force(quiet):
    """Distinct gt: F, G and the arrays FG, U, F2, H1 against the full AO tensors of the RI
    basis, assembled with the masked orbitals and one operator per group pair."""
    from gemqc.f12xg import ExactF12XGIntegrals
    a, ri, groups = _he2_groups()
    gem = XGGeminal([0.7, 1.9], 6)
    ints = ExactF12XGIntegrals(ri, 0, 2, gem, groups, 1e-14)
    C, bri = ri.C, ri.basis
    Ca = C[:, :2]
    M = []
    for G in range(2):
        m = np.zeros(bri.nbf)
        m[ri.obs_idx] = groups.mask(G)
        M.append(Ca * m[:, None])
    S = np.asarray(_core.overlap(bri))
    H = [C @ (ri.hj @ (C.T @ (S @ Mg))) for Mg in M]
    V4 = {}

    def v4(op):
        key = (op.kind, op.zeta, op.scale, tuple(op.gtg), tuple(op.gtg2))
        if key not in V4:
            V4[key] = _core.two_body_4c(op, bri)
        return V4[key]

    def T(G, H_):
        return (min(G, H_), max(G, H_))
    gg = [(G, H_) for G in range(2) for H_ in range(2)]
    Fref = np.zeros((2, 2) + (C.shape[1],) * 2)
    FG, U, F2, H1 = (np.zeros((2,) * 4) for _ in range(4))
    for G, H_ in gg:
        ops = gem.operators(G, H_)
        Fref += np.einsum("abcd,aP,bi,cQ,dj->ijPQ", v4(ops["f12"]), C, M[G], C, M[H_],
                          optimize=True)
        FG += np.einsum("abcd,am,bk,cn,dl->klmn", v4(ops["f12/r12"]), Ca, M[G], Ca, M[H_],
                        optimize=True)
        for G1, H1_ in gg:
            prod = gem.product_operators(T(G1, H1_), T(G, H_))
            U += np.einsum("abcd,am,bk,cn,dl->klmn", v4(prod["grad f12 . grad f12"]),
                           M[G1], M[G], M[H1_], M[H_], optimize=True)
            X = v4(prod["f12 x f12"])
            F2 += np.einsum("abcd,am,bk,cn,dl->klmn", X, M[G1], M[G], M[H1_], M[H_],
                            optimize=True)
            H1 += np.einsum("abcd,am,bk,cn,dl->klmn", X, H[G1], M[G], M[H1_], M[H_],
                            optimize=True)
    Gref = np.einsum("abcd,ai,bP,cj,dQ->ijPQ", v4(_core.TwoBodyOperator("coulomb")), Ca, C,
                     Ca, C, optimize=True)
    nmo = ri.nmo
    for i, j in ((0, 0), (1, 0), (1, 1)):
        p = ints.pair(i, j)
        assert np.abs(p["F"] - Fref[i, j]).max() < 1e-12
        assert np.abs(p["G"][:, :nmo] - Gref[i, j][:, :nmo]).max() < 1e-12
        assert np.abs(p["G"][:2, nmo:] - Gref[i, j][:2, nmo:]).max() < 1e-12
    for name, ref in (("FG", FG), ("U", U), ("F2", F2), ("H1", H1)):
        assert np.abs(getattr(ints, name) - ref).max() < 1e-12, name
    # <g_kl|g_mn> is symmetric in the two pairs
    assert np.abs(F2 - np.swapaxes(np.swapaxes(F2, 0, 2), 1, 3)).max() < 1e-12
    # the groups matter: F differs from the single-gamma geminals of both groups
    from gemqc.f12 import ExactF12Integrals
    for g in gem.gammas:
        std = ExactF12Integrals(ri, 0, 2, Geminal.make(g, 6), 1e-14)
        assert np.abs(std.pair(1, 0)["F"] - ints.pair(1, 0)["F"]).max() > 1e-3
        std.close()
    ints.close()


# --- task mp2-f12xg-pilot ---------------------------------------------------------------

WATER_INP = ("  w:\n    basis: cc-pVDZ-F12\n    geometry: |\n"
             "      O     0.0000000    0.0000000    0.3935881\n"
             "      H     0.7571222    0.0000000   -0.1967941\n"
             "      H    -0.7571222    0.0000000   -0.1967941\n")


def _run(tmp_path, systems, tasks):
    import gemqc
    inp = tmp_path / "job.yaml"
    inp.write_text("systems:\n" + systems + "tasks:\n" + tasks)
    return [r for _, r in gemqc.run(inp, quiet=True)]


KEYS = ("e_f12", "e_f12_3c", "e_f12_3sc", "e_f12_singlet", "e_f12_triplet",
        "mp2_correlation_energy", "e_cabs_singles", "energy")


def test_task_equal_gammas(tmp_path):
    """gt_1 = gt_2: the task reproduces mp2-f12 (3C, 3*C; NOZ, HY1), for core/valence and
    atom groups."""
    res = _run(tmp_path, WATER_INP,
               "  - {system: w, task: scf, convergence: 1e-10}\n"
               "  - {task: mp2-f12, gamma: 0.9}\n"
               "  - {task: mp2-f12xg-pilot, gammas: [0.9, 0.9], bf_groups: core_valence}\n"
               "  - {task: mp2-f12, gamma: 1.1, noz: true, hy1: true, approximation: 3*c}\n"
               "  - {task: mp2-f12xg-pilot, gammas: [1.1, 1.1], bf_groups: {atoms: [H]},"
               " noz: true, hy1: true, approximation: 3*c}\n")
    for std, xg in ((res[1], res[2]), (res[3], res[4])):
        assert xg["method"] == "MP2-F12XG" and xg["approximation"] == std["approximation"]
        for k in KEYS:
            assert xg[k] == pytest.approx(std[k], abs=1e-11), k
    assert res[2]["group_sizes"] == [1, 47] and res[4]["group_sizes"] == [18, 30]


def test_task_additivity_he_ne(tmp_path):
    """He ... Ne (50 Angstrom), groups by atom with gt_He != gt_Ne: the sum of mp2-f12 for
    He (gamma = gt_He) and Ne (gamma = gt_Ne)."""
    basis = "    basis: aug-cc-pVDZ\n    symmetry: nosym\n"
    systems = (f"  hene:\n{basis}    geometry: |\n      He 0 0 0\n      Ne 0 0 50\n"
               f"  he:\n{basis}    geometry: 'He 0 0 0'\n"
               f"  ne:\n{basis}    geometry: 'Ne 0 0 0'\n")
    res = _run(tmp_path, systems,
               "  - {system: hene, task: scf, convergence: 1e-10}\n"
               "  - {task: mp2-f12xg-pilot, gammas: [1.6, 0.8], bf_groups: {atoms: [He]}}\n"
               "  - {system: he, task: scf, convergence: 1e-10}\n"
               "  - {task: mp2-f12, gamma: 1.6}\n"
               "  - {system: ne, task: scf, convergence: 1e-10}\n"
               "  - {task: mp2-f12, gamma: 0.8}\n")
    xg, he, ne = res[1], res[3], res[5]
    for k in KEYS:
        assert xg[k] == pytest.approx(he[k] + ne[k], abs=1e-9), k
    # a single gamma for both atoms is different
    assert abs(he["e_f12"] - ne["e_f12"]) > 1e-3


def test_task_core_valence_output(tmp_path):
    """Distinct gammas with core/valence groups: results, JSON and HDF5 entries; input
    errors."""
    import h5py
    import gemqc
    res = _run(tmp_path, WATER_INP,
               "  - {system: w, task: scf, convergence: 1e-10}\n"
               "  - {task: mp2-f12xg-pilot, gammas: [2.5, 0.9], bf_groups: core_valence,"
               " frozen: off, pair_energies: true}\n")
    xg = res[1]
    assert xg["gammas"] == [2.5, 0.9] and xg["group_sizes"] == [1, 47]
    assert xg["bf_groups"].startswith("core_valence")
    assert xg["correlation_energy"] == pytest.approx(xg["mp2_correlation_energy"]
                                                     + xg["e_f12"], abs=1e-12)
    with h5py.File(tmp_path / "job.h5") as f:
        g = f["w/mp2_f12xg"]
        assert list(g.attrs["gammas"]) == [2.5, 0.9]
        assert np.sum(g.attrs["bf_groups"] == 1) == 1
        assert np.asarray(g["pair_energies_3c"]).sum() == pytest.approx(xg["e_f12"],
                                                                        abs=1e-12)
    scf = "  - {system: w, task: scf}\n"
    for bad in ("{task: mp2-f12xg-pilot, gammas: [1.0], bf_groups: core_valence}",
                "{task: mp2-f12xg-pilot, gammas: [1.0, 2.0]}",
                "{task: mp2-f12xg-pilot, gammas: [1.0, 2.0], bf_groups: {atoms: [7]}}"):
        with pytest.raises(gemqc.InputError):
            _run(tmp_path, WATER_INP, scf + "  - " + bad + "\n")


# --- density-fitted F12XG integrals -----------------------------------------------------

def _he2_fit(ri):
    from gemqc.basislib import build_basis
    mol = _core.Molecule()
    for z in ([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]):
        mol.add_atom(2, z)
    return build_basis(mol, "aug-cc-pVDZ-C")


@pytest.mark.parametrize("fit", [6, "exact"])
def test_df_equal_gammas(fit, quiet):
    """gt_1 = gt_2: DF-F12XG pair quantities equal standard DF-MP2-F12 for any groups."""
    from gemqc.f12 import F12Integrals, PairIntermediates
    from gemqc.f12xg import DFF12XGIntegrals
    a, ri, groups = _he2_groups()
    fb = _he2_fit(ri)
    std = F12Integrals(ri, 0, 2, fb, Geminal.make(1.2, fit))
    xg = DFF12XGIntegrals(ri, 0, 2, fb, XGGeminal([1.2, 1.2], fit), groups)
    for i, j in ((0, 0), (1, 0), (1, 1)):
        ps, px = std.pair(i, j), xg.pair(i, j)
        assert np.abs(ps["F"] - px["F"]).max() < 1e-12
        assert np.abs(ps["G"] - px["G"]).max() < 1e-12
    for noz, hy1 in ((False, False), (True, True)):
        ps, px = (PairIntermediates(x, ri, noz, hy1) for x in (std, xg))
        for i, j in ((0, 0), (1, 0), (1, 1)):
            ds, dx = ps.pair(i, j), px.pair(i, j)
            for key in ("V", "X", "B3sC", "B3C", "C", "K"):
                assert np.abs(ds[key] - dx[key]).max() < 1e-12, key
    std.close()
    xg.close()


def test_df_vs_exact_xg(quiet):
    """Distinct gt: the DF error of F12XG (against exact F12XG) is within the DF errors of
    standard MP2-F12 for the two group exponents (same fitting basis): relative errors of
    V, X, B and the V and U blocks."""
    from gemqc.f12 import ExactF12Integrals, F12Integrals, PairIntermediates
    from gemqc.f12xg import DFF12XGIntegrals, ExactF12XGIntegrals
    a, ri, groups = _he2_groups()
    fb = _he2_fit(ri)
    pairs = ((0, 0), (1, 0), (1, 1))

    def errors(ex, df):
        pe, pd = PairIntermediates(ex, ri), PairIntermediates(df, ri)
        out = {}
        for k in ("V", "X", "B3C"):
            out[k] = max(np.abs(pe.pair(*p)[k] - pd.pair(*p)[k]).max()
                         / np.abs(pe.pair(*p)[k]).max() for p in pairs)
        for k in ("f12/r12", "(grad f12)^2"):
            out[k] = max(np.abs(ex.pair(*p)[k] - df.pair(*p)[k]).max()
                         / np.abs(ex.pair(*p)[k]).max() for p in pairs)
        ex.close()
        df.close()
        return out
    gem = XGGeminal([0.7, 1.9], 6)
    xg = errors(ExactF12XGIntegrals(ri, 0, 2, gem, groups, 1e-14),
                DFF12XGIntegrals(ri, 0, 2, fb, gem, groups))
    std = [errors(ExactF12Integrals(ri, 0, 2, Geminal.make(g, 6), 1e-14),
                  F12Integrals(ri, 0, 2, fb, Geminal.make(g, 6))) for g in gem.gammas]
    for k, e in xg.items():
        assert e < 1.5 * max(s_[k] for s_ in std), k


WATER_ADZ = WATER_INP.replace("cc-pVDZ-F12", "aug-cc-pVDZ")


def test_df_task(tmp_path):
    """df-mp2-f12xg-pilot: equal gammas reproduce df-mp2-f12; distinct gammas agree with
    the exact pilot within the DF error (as df-mp2-f12 against mp2-f12)."""
    res = _run(tmp_path, WATER_ADZ,
               "  - {system: w, task: scf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12, gamma: 1.1}\n"
               "  - {task: df-mp2-f12xg-pilot, gammas: [1.1, 1.1], bf_groups: core_valence}\n"
               "  - {task: df-mp2-f12xg-pilot, gammas: [2.5, 0.9], bf_groups: core_valence,"
               " frozen: off}\n"
               "  - {task: mp2-f12xg-pilot, gammas: [2.5, 0.9], bf_groups: core_valence,"
               " frozen: off}\n")
    std, eq, df, ex = res[1:]
    assert eq["method"] == "DF-MP2-F12XG" and eq["density_fitting"] == std["density_fitting"]
    for k in KEYS:
        assert eq[k] == pytest.approx(std[k], abs=1e-11), k
    # DF error of all-electron mp2-f12 for water/aug-cc-pVDZ: e_f12 4e-5 (gamma 2.5),
    # 9e-5 (gamma 0.9); here 1.0e-4
    for k in ("e_f12", "e_f12_3sc", "mp2_correlation_energy"):
        assert df[k] == pytest.approx(ex[k], abs=2e-4), k
    assert df[k] != ex[k]


def test_df_task_additivity_he_ne(tmp_path):
    """df-mp2-f12xg-pilot for He ... Ne with groups by atom: the sum of df-mp2-f12 for He
    and Ne with their own exponents."""
    basis = ("    basis: aug-cc-pVDZ\n    symmetry: nosym\n"
             "    jk_auxbasis: def2-universal-JK\n")  # aug-cc-pVTZ-JK has no He
    systems = (f"  hene:\n{basis}    geometry: |\n      He 0 0 0\n      Ne 0 0 50\n"
               f"  he:\n{basis}    geometry: 'He 0 0 0'\n"
               f"  ne:\n{basis}    geometry: 'Ne 0 0 0'\n")
    res = _run(tmp_path, systems,
               "  - {system: hene, task: scf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12xg-pilot, gammas: [1.6, 0.8], bf_groups: {atoms: [He]}}\n"
               "  - {system: he, task: scf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12, gamma: 1.6}\n"
               "  - {system: ne, task: scf, convergence: 1e-10}\n"
               "  - {task: df-mp2-f12, gamma: 0.8}\n")
    xg, he, ne = res[1], res[3], res[5]
    # standard df-mp2-f12 is itself additive only to 2e-7 here (e_f12; CABS singles 4e-8):
    # inherent to Coulomb-metric fitting, decays as 1/R (test_mp2_f12.py); exact: 1e-13
    for k in KEYS:
        assert xg[k] == pytest.approx(he[k] + ne[k], abs=1e-6), k


def test_core_valence_per_atom_basis():
    """Basis per atom: the valence reference is derived per atom (canonical name)."""
    from gemqc.basislib import build_basis
    from gemqc.f12xg import assign_bf_groups, valence_reference_name
    mol = _mol(WATER)
    zs = list(mol.atomic_numbers)
    assert valence_reference_name("cc-pCVDZ-F12;H=cc-pVDZ-F12", zs) == "cc-pVDZ-F12"
    assert valence_reference_name("cc-pVDZ-F12", zs) is None
    assert valence_reference_name("cc-pVTZ;O=cc-pwCVTZ", zs) == "cc-pVTZ"
    bs = build_basis(mol, {"default": "cc-pVDZ-F12", "O": "cc-pCVDZ-F12"})
    g = assign_bf_groups(mol, bs, "core_valence")
    assert "4 functions not in cc-pVDZ-F12)" in g.how
    assert g.summary(mol, bs)[1].startswith("group 1: 5 functions (O1:5)")
