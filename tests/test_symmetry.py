"""Point-group detection, standard orientation, AO operations, orbital labels."""

from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.symmetry import (_apply, _rotation, analyze, ao_operations, orbital_irreps,
                            orbital_labels, symmetrize_molecule)

DATA = Path(__file__).parent / "data"


def ring(n, r, z=0.0, phase=0.0):
    return [[r * np.cos(2 * np.pi * k / n + phase), r * np.sin(2 * np.pi * k / n + phase), z]
            for k in range(n)]


t, p = 1.2, (1 + 5 ** 0.5) / 2
ico = [[0, a, b * p] for a in (1, -1) for b in (1, -1)]
ico = ico + [[v[2], v[0], v[1]] for v in ico] + [[v[1], v[2], v[0]] for v in ico]
MOLECULES = {  # name: (Z, coordinates / bohr, full group, computational group)
    "H2O": ([8, 1, 1], [[0, 0, 0.12], [1.43, 0, -0.98], [-1.43, 0, -0.98]], "C2v", "C2v"),
    "NH3": ([7, 1, 1, 1], [[0, 0, 0.2]] + ring(3, 1.8, -0.5), "C3v", "Cs"),
    "CH4": ([6, 1, 1, 1, 1], [[0, 0, 0], [t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]],
            "Td", "C2v"),
    "C2H4": ([6, 6, 1, 1, 1, 1], [[0, 0, 1.26], [0, 0, -1.26], [0, 1.75, 2.33],
                                  [0, -1.75, 2.33], [0, 1.75, -2.33], [0, -1.75, -2.33]],
             "D2h", "D2h"),
    "C6H6": ([6] * 6 + [1] * 6, ring(6, 2.64) + ring(6, 4.68), "D6h", "D2h"),
    "CO2": ([6, 8, 8], [[0, 0, 0], [0, 0, 2.2], [0, 0, -2.2]], "Dinfh", "D2h"),
    "HCN": ([1, 6, 7], [[0, 0, -2], [0, 0, 0], [0, 0, 2.2]], "Cinfv", "C2v"),
    "C2H6": ([6, 6] + [1] * 6, [[0, 0, 1.45], [0, 0, -1.45]] + ring(3, 1.9, 2.1)
             + ring(3, 1.9, -2.1, np.pi / 3), "D3d", "C2h"),
    "allene": ([6, 6, 6, 1, 1, 1, 1], [[0, 0, 0], [0, 0, 2.5], [0, 0, -2.5], [1.75, 0, 3.5],
                                       [-1.75, 0, 3.5], [0, 1.75, -3.5], [0, -1.75, -3.5]],
               "D2d", "C2v"),
    "SF6": ([16] + [9] * 6, [[0, 0, 0]] + [list(3 * v) for v in np.vstack([np.eye(3), -np.eye(3)])],
            "Oh", "D2h"),
    "B12": ([5] * 12, [list(1.6 * np.array(v)) for v in ico], "Ih", "D2h"),
    "C5H5": ([6] * 5 + [1] * 5, ring(5, 2.2) + ring(5, 4.2), "D5h", "C2v"),
    "H2O2": ([8, 8, 1, 1], [[1.3, 0.3, 0], [-1.3, -0.3, 0], [1.6, 0.9, 1.7], [-1.6, -0.9, 1.7]],
             "C2", "C2"),
    "atom": ([10], [[0.3, 0.2, 0.1]], "Kh", "D2h"),
}


def random_frame(X, seed=5):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    return np.asarray(X, float) @ Q.T + [0.7, -1.1, 0.4]


@pytest.mark.parametrize("name", MOLECULES)
def test_point_groups(name):
    Z, X, full, comp = MOLECULES[name]
    sym, xyz = analyze(Z, random_frame(X))
    assert (sym.point_group, sym.group) == (full, comp)
    assert sym.max_displacement < 1e-10


def test_standard_orientation_and_symmetrization():
    Z, X, _, _ = MOLECULES["NH3"]
    X = random_frame(X)
    X[1] += 2e-4  # small distortion, within tolerance
    sym, xyz = analyze(Z, X)
    assert sym.point_group == "C3v" and 1e-5 < sym.max_displacement < 1e-3
    assert np.allclose(xyz[:, 2][sym.atom_map[1]], -xyz[:, 2])  # sigma(xy) exact
    # water: C2 axis z, molecular plane yz
    Z, X, _, _ = MOLECULES["H2O"]
    sym, xyz = analyze(Z, random_frame(X))
    assert np.allclose(xyz[:, 0], 0) and np.allclose(xyz[0, :2], 0)
    assert np.allclose(sym.to_standard(random_frame(X)), xyz)


def test_modify():
    Z, X, _, _ = MOLECULES["H2O"]  # C2 along z, plane xz
    dist = lambda Y: np.linalg.norm(Y[:, None] - Y[None], axis=2)
    sym, xyz = analyze(Z, X, modify="off")  # input frame carries C2v
    assert sym.group == "C2v" and xyz is None and sym.tolerance == 1e-10
    sym, xyz = analyze(Z, random_frame(X), modify="off")
    assert sym.group == "C1" and xyz is None
    with pytest.raises(ValueError, match="input frame"):
        analyze(Z, random_frame(X), request="c2v", modify="off")
    sym, xyz = analyze(Z, random_frame(X), modify="align")
    assert sym.group == "C2v" and np.allclose(dist(xyz), dist(np.array(X)), atol=1e-12)
    # distortion within the adapt tolerance: align keeps it (lower symmetry)
    Z, X, _, _ = MOLECULES["NH3"]
    X = random_frame(X)
    X[1] += 2e-4
    sym, xyz = analyze(Z, X, modify="align")
    assert sym.group == "C1"
    sym, xyz = analyze(Z, X, modify="adapt")
    assert sym.group == "Cs" and sym.max_displacement > 1e-5


def test_requested_groups():
    Z, X, _, _ = MOLECULES["C2H4"]
    for req in ("c2v", "c2h", "d2", "cs", "c2", "ci"):
        sym, _ = analyze(Z, random_frame(X), request=req)
        assert sym.group.lower() == req
    sym, xyz = analyze(Z, random_frame(X), request="nosym")
    assert sym.group == "C1" and xyz is None
    Z, X, _, _ = MOLECULES["H2O"]
    with pytest.raises(ValueError, match="does not have"):
        analyze(Z, X, request="d2h")


def test_atom_types():
    """Atom types (e.g. basis sets per atom) lower the symmetry: equivalent atoms only
    with equal types; types following Z change nothing."""
    Z, X, _, _ = MOLECULES["C2H4"]
    for types, full, comp in (([0, 0, 1, 1, 1, 1], "D2h", "D2h"),   # per element
                              ([0, 0, 1, 2, 1, 2], "C2v", "C2v"),   # one side of C=C
                              ([0, 0, 1, 1, 2, 2], "C2v", "C2v"),   # one CH2 group
                              ([0, 0, 1, 2, 2, 1], "C2h", "C2h"),   # trans
                              ([0, 1, 2, 2, 2, 2], "C2v", "C2v"),   # one C
                              ([0, 0, 1, 2, 3, 3], "Cs", "Cs")):
        sym, xyz = analyze(Z, random_frame(X), types=types)
        assert (sym.point_group, sym.group) == (full, comp)
        assert sym.nuclear_point_group == ("" if full == "D2h" else "D2h")
        for row in sym.atom_map:
            assert all(types[a] == types[b] for a, b in enumerate(row))
    with pytest.raises(ValueError, match="basis sets per atom do not allow d2h"):
        analyze(Z, X, request="d2h", types=[0, 0, 1, 2, 1, 2])
    Z, X, _, _ = MOLECULES["H2O"]
    with pytest.raises(ValueError, match="does not have"):  # not the basis: the nuclei
        analyze(Z, X, request="d2h", types=[0, 1, 2])


@pytest.mark.parametrize("name, pure", [("C6H6", True), ("CH4", True), ("CH4", False)])
def test_ao_operations_commute(name, pure):
    Z, X, _, _ = MOLECULES[name]
    mol = gemqc.Molecule()
    for z, r in zip(Z, random_frame(X)):
        mol.add_atom(z, list(r))
    sym = symmetrize_molecule(mol)
    lib = {z: [gemqc.ShellDef(l, pure, [0.9 + 0.3 * l], [1.0]) for l in range(4)] for z in set(Z)}
    bs = gemqc.BasisSet(mol, lib)
    S = gemqc.overlap(bs)
    h = gemqc.kinetic(bs) + gemqc.nuclear_attraction(bs, mol)
    for rep in ao_operations(bs, sym):
        P = _apply(rep, np.eye(bs.nbf))
        assert np.allclose(P @ S @ P.T, S, atol=1e-12)
        assert np.allclose(P @ h @ P.T, h, atol=1e-11)


def test_orbital_labels():
    _core.set_output_enabled(False)
    try:
        # water (input in the xz plane, kept as in Molpro): labels, occupation 3 1 1 0
        mol = gemqc.read_xyz(DATA / "h2o.xyz")
        sym = symmetrize_molecule(mol)
        bs = gemqc.build_basis(mol, "def2-SVP")
        wf = gemqc.rhf(mol, bs)
        irr, dev = orbital_irreps(wf.C, wf.eps, gemqc.overlap(bs), bs, sym)
        assert orbital_labels(irr)[:5] == ["1.1", "2.1", "1.2", "3.1", "1.3"] and dev < 1e-8
        # methane: t2 set splits into a1 + b1 + b2 of C2v
        Z, X, _, _ = MOLECULES["CH4"]
        mol = gemqc.Molecule()
        for z, r in zip(Z, random_frame(X)):
            mol.add_atom(z, list(r))
        sym = symmetrize_molecule(mol)
        bs = gemqc.build_basis(mol, "def2-SVP")
        wf = gemqc.rhf(mol, bs)
        irr, dev = orbital_irreps(wf.C, wf.eps, gemqc.overlap(bs), bs, sym)
        assert sorted(irr[2:5]) == [0, 1, 2] and (irr >= 0).all() and dev < 1e-6
    finally:
        _core.set_output_enabled(True)


def test_driver_symmetry(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "s.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "  c1: {coordinates: h2o.xyz, basis: def2-SVP, symmetry: nosym}\n"
                   "tasks:\n  - {system: w, task: scf, convergence: 1e-8}\n"
                   "  - {system: c1, task: scf, convergence: 1e-8}\n")
    (_, r1), (_, r2) = gemqc.run(inp, quiet=True)
    assert r1["closed_shells_per_irrep"] == [3, 1, 1, 0]
    assert r1["energy"] == pytest.approx(r2["energy"], abs=1e-10)
    assert r1["energy"] == pytest.approx(-75.960841436478, abs=1e-9)


# --- stage 2: symmetry-blocked SCF ---------------------------------------------

def symmetric_system(name, basis="def2-SVP"):
    Z, X, _, _ = MOLECULES[name]
    mol = gemqc.Molecule(name)
    for z, r in zip(Z, random_frame(X)):
        mol.add_atom(z, list(r))
    sym = symmetrize_molecule(mol)
    return mol, sym, gemqc.build_basis(mol, basis)


def test_so_basis():
    from gemqc.symmetry import so_basis
    mol, sym, bs = symmetric_system("C6H6")
    U = so_basis(bs, sym)
    assert sum(u.shape[1] for u in U) == bs.nbf
    full = np.hstack(U)
    assert np.allclose(full.T @ full, np.eye(bs.nbf), atol=1e-12)  # orthonormal, complete
    S = gemqc.overlap(bs)
    for k in range(len(U)):
        for l in range(k):
            assert np.abs(U[k].T @ S @ U[l]).max() < 1e-12      # irreps do not mix


@pytest.mark.parametrize("name", ["H2O", "CH4", "C2H4"])
def test_blocked_scf_matches_c1(name):
    from gemqc.symmetry import scf_symmetry
    _core.set_output_enabled(False)
    try:
        mol, sym, bs = symmetric_system(name)
        opt = gemqc.ScfOptions()
        opt.convergence = 1e-8
        wf1 = gemqc.rhf(mol, bs, opt)
        wfs = gemqc.rhf(mol, bs, opt, None, scf_symmetry(bs, sym))
    finally:
        _core.set_output_enabled(True)
    assert wfs.energy == pytest.approx(wf1.energy, abs=1e-10)
    assert wfs.group == sym.group
    # exact irreps agree with the projector analysis of the C1 orbitals (occupied)
    irr, _ = orbital_irreps(wf1.C, wf1.eps, gemqc.overlap(bs), bs, sym)
    assert sorted(irr[:wf1.n_occ]) == sorted(wfs.irreps[:wfs.n_occ])


def test_fixed_and_frozen_occupation(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "o.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "tasks:\n  - {system: w, task: scf, occupation: [3, 1, 1, 0], save: false}\n"
                   "  - {task: scf, freeze_occupation: 1, guess: sad, save: false}\n"
                   "  - {task: scf, occupation: [2, 1, 2, 0], guess: sad, save: false}\n")
    (_, r1), (_, r2), (_, r3) = gemqc.run(inp, quiet=True)
    assert r1["energy"] == pytest.approx(-75.960841436478, abs=1e-8)
    assert r2["energy"] == pytest.approx(r1["energy"], abs=1e-8)
    assert r3["closed_shells_per_irrep"] == [2, 1, 2, 0]   # excited determinant
    assert r3["energy"] > r1["energy"] + 0.1
    for occ, msg in (("[3, 1, 1]", "4 numbers"), ("[3, 1, 1, 1]", "sums to 6")):
        inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                       f"tasks:\n  - {{system: w, task: scf, occupation: {occ}, save: false}}\n")
        with pytest.raises(gemqc.InputError, match=msg):
            gemqc.run(inp, quiet=True)


# --- stage 3: skeleton Fock build ------------------------------------------------

@pytest.mark.parametrize("name", ["H2O", "NH3", "C2H4", "C2H6", "H2O2", "CH4"])
def test_skeleton_jk(name):
    """Symmetry-unique quartets + symmetrization == full J/K (symmetric D)."""
    mol, sym, bs = symmetric_system(name)
    reps = ao_operations(bs, sym)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((bs.nbf, bs.nbf))
    A = A + A.T
    D = sum(_apply(r, _apply(r, A).T).T for r in reps) / len(reps)  # totally symmetric
    (J0,), (K0,) = gemqc.FockBuilder(bs, 1e-14).compute([D])
    fb = gemqc.FockBuilder(bs, 1e-14)
    fb.set_symmetry([list(p) for p, _ in reps], [list(s) for _, s in reps])
    (J,), (K,) = fb.compute([D])
    assert fb.symmetry_order == sym.order
    assert np.allclose(J, J0, atol=1e-12) and np.allclose(K, K0, atol=1e-12)
    full = gemqc.FockBuilder(bs, 1e-14)
    full.compute([D])
    assert fb.quartets_computed < full.quartets_computed


@pytest.mark.parametrize("name", ["H2O", "C2H4", "H2O2"])
def test_skeleton_jk_irreps(name):
    """Skeleton J/K for general (non-symmetric) densities of every irrep."""
    from gemqc.symmetry import character_table
    mol, sym, bs = symmetric_system(name)
    reps = ao_operations(bs, sym)
    chars = character_table(sym)
    rng = np.random.default_rng(5)
    A = rng.standard_normal((bs.nbf, bs.nbf))
    # projection onto irrep k: (1/h) sum_g chi_k(g) R_g A R_g^T
    Ds = [sum(c * _apply(r, _apply(r, A).T).T for c, r in zip(chi, reps)) / len(reps)
          for chi in chars]
    J0, K0 = gemqc.FockBuilder(bs, 1e-14).compute(Ds, symmetric=False)
    fb = gemqc.FockBuilder(bs, 1e-14)
    fb.set_symmetry([list(p) for p, _ in reps], [list(s) for _, s in reps], chars)
    J, K = fb.compute(Ds, symmetric=False, irreps=list(range(len(chars))))
    for k in range(len(chars)):
        assert np.allclose(J[k], J0[k], atol=1e-12) and np.allclose(K[k], K0[k], atol=1e-12)
    with pytest.raises(ValueError):
        fb.compute(Ds[:1], irreps=[len(chars)])


def test_skeleton_scf(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "k.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-TZVPP}\n"
                   "tasks:\n  - {system: w, task: scf, convergence: 1e-8, save: false}\n"
                   "  - {task: scf, convergence: 1e-8, skeleton: false, save: false}\n")
    (_, r1), (_, r2) = gemqc.run(inp, quiet=True)
    assert r1["energy"] == pytest.approx(r2["energy"], abs=1e-10)
    assert r1["energy"] == pytest.approx(-76.062277305532, abs=1e-9)


def test_long_orbital_printout(tmp_path):
    import contextlib, io
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    (tmp_path / "nh2.xyz").write_text((DATA / "nh2.xyz").read_text())
    inp = tmp_path / "l.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "  r: {coordinates: nh2.xyz, basis: def2-SVP, multiplicity: 2}\n"
                   "tasks:\n  - {system: w, task: scf, orbitals: long, save: false}\n"
                   "  - {system: w, task: scf, orbitals: full, save: false}\n"
                   "  - {system: r, task: uhf, orbitals: long, save: false}\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        (_, res), _, _ = gemqc.run(inp)
    tables = buf.getvalue().split("in energetic order")[1:]

    def table_rows(text):
        rows = []
        for line in text.split("\n\n", 2)[1].splitlines():
            f = line.split()
            if f and f[0].isdigit():
                rows.append(f)
        return rows
    long_, full, uhf = (table_rows(t) for t in tables)
    assert len(long_) == 8 and len(full) == 24   # long: up to the third unoccupied orbital
    assert long_[4][:3] == ["5", "1.3", "B2"] and float(long_[4][4]) == 2.0
    assert long_[4][5:] == ["0.980", "O", "p"]  # character: O 2p lone pair
    assert [float(r[4]) for r in long_[5:]] == [0.0] * 3
    assert res["orbital_occupations"][:6].tolist() == [2, 2, 2, 2, 2, 0]
    # UHF: alpha and beta in one table, spin column, energetic order
    assert [r[2] for r in uhf[:4]] == ["a", "b", "a", "b"] and len(uhf) == 12
    energies = [float(r[4]) for r in uhf]
    assert energies == sorted(energies)
    assert sum(float(r[5]) == 0.0 for r in uhf) == 3
