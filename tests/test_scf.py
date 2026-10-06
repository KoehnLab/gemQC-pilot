"""Direct J/K build, RHF, SAD guess, restart, resources."""

import json
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.guess import atomic_configuration

DATA = Path(__file__).parent / "data"
E_WATER_SVP = -75.960841436  # RHF/def2-SVP, tests/data/h2o.xyz (numpy reference SCF)


@pytest.fixture(scope="module")
def water():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    return mol, gemqc.build_basis(mol, "def2-SVP")


@pytest.fixture(autouse=True)
def quiet():
    _core.set_output_enabled(False)
    yield
    _core.set_output_enabled(True)
    _core.set_memory(0)
    _core.set_threads(0)


def test_eri_symmetry(water):
    g = gemqc.eri(water[1])
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.allclose(g, g.transpose(perm), atol=1e-14)


@pytest.mark.parametrize("symmetric", [True, False])
def test_jk_vs_eri(water, symmetric):
    _, bs = water
    g = gemqc.eri(bs)
    rng = np.random.default_rng(7)
    ds = [rng.standard_normal((bs.nbf, bs.nbf)) for _ in range(2)]
    if symmetric:
        ds = [d + d.T for d in ds]
    else:  # general, plus symmetric and antisymmetric (detected, 4-term K path)
        ds += [ds[0] + ds[0].T, ds[1] - ds[1].T]
    J, K = gemqc.FockBuilder(bs, 1e-14).compute(ds, symmetric=symmetric)
    for d, j, k in zip(ds, J, K):
        assert np.allclose(j, np.einsum("pqrs,rs->pq", g, d), atol=1e-12)
        assert np.allclose(k, np.einsum("prqs,rs->pq", g, d), atol=1e-12)


@pytest.mark.parametrize("atoms", [
    [(6, 0.0), (6, 3.0)],      # tight s / d on distant atoms: (ab|ab) is dropped by
    [(1, 0.0), (1, 4.0)],      # libint's primitive pre-screening -> Schwarz bound 0
    [(6, 0.0), (1, 3.0), (8, 6.5)],
])
def test_jk_distant_atoms(atoms):
    """Regression: significant pairs of distant atoms must not be lost."""
    mol = gemqc.Molecule()
    for z, r in atoms:
        mol.add_atom(z, [0.3 * r, 0.1, r / 0.529177210903])
    bs = gemqc.build_basis(mol, "def2-SVP")
    g = gemqc.eri(bs)
    rng = np.random.default_rng(1)
    D = rng.standard_normal((bs.nbf, bs.nbf))
    D = D + D.T
    (J,), (K,) = gemqc.FockBuilder(bs, 1e-16).compute([D])
    assert np.allclose(J, np.einsum("pqrs,rs->pq", g, D), atol=1e-13)
    assert np.allclose(K, np.einsum("prqs,rs->pq", g, D), atol=1e-13)


def numpy_rhf(mol, bs):
    """Plain reference SCF with the full ERI tensor (damped iterations)."""
    g = gemqc.eri(bs)
    S = gemqc.overlap(bs)
    h = gemqc.kinetic(bs) + gemqc.nuclear_attraction(bs, mol)
    s, U = np.linalg.eigh(S)
    X = U / np.sqrt(s)
    nocc = mol.n_electrons // 2
    D = np.zeros_like(S)
    for _ in range(300):
        F = h + np.einsum("pqrs,rs->pq", g, D) - 0.5 * np.einsum("prqs,rs->pq", g, D)
        C = X @ np.linalg.eigh(X.T @ F @ X)[1]
        D_new = 2 * C[:, :nocc] @ C[:, :nocc].T
        if np.abs(D_new - D).max() < 1e-10:
            break
        D = 0.5 * (D + D_new)
    return 0.5 * np.sum(D * (h + F)) + mol.nuclear_repulsion()


def test_rhf_water(water):
    mol, bs = water
    e_ref = numpy_rhf(mol, bs)
    assert e_ref == pytest.approx(E_WATER_SVP, abs=1e-8)
    opt = gemqc.ScfOptions()
    opt.convergence, opt.energy_convergence = 1e-8, 1e-11
    for guess in (None, gemqc.sad_guess(mol, bs)):
        wf = gemqc.rhf(mol, bs, opt, guess)
        assert wf.converged
        assert wf.energy == pytest.approx(e_ref, abs=1e-9)
    # orbitals: orthonormal, density consistent, Brillouin (occ-virt Fock block) ~ 0
    S = gemqc.overlap(bs)
    C = wf.C
    assert np.allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-10)
    assert np.allclose(wf.D, 2 * C[:, :5] @ C[:, :5].T, atol=1e-8)
    assert wf.n_occ == 5 and wf.eps[4] < 0 < wf.eps[5]


def test_rhf_variants_agree(water):
    mol, bs = water
    energies = []
    for incremental, threads in ((True, 0), (False, 0), (True, 1)):
        opt = gemqc.ScfOptions()
        opt.incremental = incremental
        _core.set_threads(threads)
        energies.append(gemqc.rhf(mol, bs, opt).energy)
    assert np.ptp(energies) < 1e-9


def test_rhf_open_shell_rejected(water):
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    mol.charge, mol.multiplicity = 1, 2
    with pytest.raises(ValueError):
        gemqc.rhf(mol, water[1])


def test_sad_guess(water):
    mol, bs = water
    assert atomic_configuration(8) == {(1, 0): 2, (2, 0): 2, (2, 1): 4}
    assert atomic_configuration(24)[(3, 2)] == 5  # Cr: 4s1 3d5
    D = gemqc.sad_guess(mol, bs)
    assert np.allclose(D, D.T)
    assert np.sum(D * gemqc.overlap(bs)) == pytest.approx(10.0, abs=0.1)


def test_ecp_basis_rejected():
    mol = gemqc.Molecule()
    mol.add_atom(53, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="ECP"):
        gemqc.build_basis(mol, "def2-SVP")


def test_memory_limit(water):
    _, bs = water
    _core.set_memory(1000)  # bytes
    with pytest.raises(RuntimeError, match="memory"):
        gemqc.eri(bs)
    with pytest.raises(RuntimeError, match="memory"):
        gemqc.FockBuilder(bs).compute([np.eye(bs.nbf)])


def test_scf_task_and_restart(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "w.yaml"
    inp.write_text("systems:\n  water: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "resources: {memory: 100 MB, threads: 2}\n"
                   "tasks:\n  - {system: water, task: scf, convergence: 1e-7}\n"
                   "  - {task: scf, guess: restart}\n")
    (_, r1), (_, r2) = gemqc.run(inp, quiet=True)
    assert r1["energy"] == pytest.approx(E_WATER_SVP, abs=1e-8)
    assert r2["iterations"] <= 2  # restart from converged orbitals
    data = json.loads((tmp_path / "w.json").read_text())
    assert data["tasks"][0]["results"]["energy"] == pytest.approx(r1["energy"])
    assert (tmp_path / "w.h5").exists()


def test_auto_restart_from_smaller_basis(tmp_path):
    """guess auto: TZ run starts from the DZ orbitals of the same molecule;
    other charge or geometry is not used; explicit restart without data fails."""
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    shifted = (DATA / "h2o.xyz").read_text().replace("0.3935881", "0.4935881")
    (tmp_path / "h2o_b.xyz").write_text(shifted)
    inp = tmp_path / "a.yaml"
    inp.write_text("systems:\n"
                   "  dz: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "  tz: {coordinates: h2o.xyz, basis: def2-TZVPP}\n"
                   "  ion: {coordinates: h2o.xyz, basis: def2-SVP, charge: 2}\n"
                   "  geo: {coordinates: h2o_b.xyz, basis: def2-SVP}\n"
                   "tasks:\n  - {system: dz, task: scf}\n  - {system: tz, task: scf}\n"
                   "  - {system: ion, task: scf}\n  - {system: geo, task: scf}\n"
                   "  - {system: geo, task: scf, restart_from: dz}\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = gemqc.run(inp)
    _core.set_output_enabled(False)
    text = buf.getvalue()
    assert "restart guess from a.h5 [dz/scf], basis def2-SVP -> def2-TZVPP" in text
    assert text.count("no restart data for this molecule found") == 3  # dz, ion, geo
    assert "[dz/scf], basis def2-SVP -> def2-SVP" in text             # restart_from
    assert res[4][1]["energy"] == pytest.approx(res[3][1]["energy"], abs=1e-8)

    inp.write_text("systems:\n  tz: {coordinates: h2o.xyz, basis: def2-TZVPP}\n"
                   "tasks:\n  - {system: tz, task: scf, guess: restart, "
                   "restart_file: none.h5}\n")
    with pytest.raises(gemqc.InputError, match="no restart data"):
        gemqc.run(inp, quiet=True)


def test_restart_projection(tmp_path):
    """Orbitals stored in ANO-R0 projected onto def2-SVP."""
    from gemqc.restart import restart_guess, save_wavefunction
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    ano = gemqc.build_basis(mol, "ANO-R0")
    wf = gemqc.rhf(mol, ano)
    save_wavefunction(tmp_path / "r.h5", "water", "scf", wf)
    svp = gemqc.build_basis(mol, "def2-SVP")
    (D,) = restart_guess(tmp_path / "r.h5", "water/scf", mol, svp)
    S = gemqc.overlap(svp)
    assert np.sum(D * S) == pytest.approx(10.0, abs=1e-10)       # re-orthonormalized
    assert np.allclose(D @ S @ D, 2 * D, atol=1e-10)             # idempotent
    assert gemqc.rhf(mol, svp, guess=D).energy == pytest.approx(E_WATER_SVP, abs=1e-8)


def test_task_option_typo(tmp_path):
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    inp = tmp_path / "t.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "tasks:\n  - {system: w, task: scf, convergense: 1e-6}\n")
    with pytest.raises(gemqc.InputError, match="did you mean 'convergence'"):
        gemqc.run(inp, quiet=True)


@pytest.mark.skipif(gemqc._core.libint_info()["max_am"] < 6,
                    reason="libint built with max. angular momentum < 6")
def test_high_angular_momentum_rotation_invariance():
    """RHF energy with h and i functions (on F) is invariant under a rigid rotation
    (one- and two-electron integrals, solid harmonics)."""
    energies = []
    rot = np.linalg.qr(np.random.default_rng(2).standard_normal((3, 3)))[0]
    for R in (np.eye(3), rot):
        mol = gemqc.Molecule("hf")
        for z, r in ((9, [0.0, 0.0, 0.0]), (1, [0.3, 0.4, 1.6])):
            mol.add_atom(z, list(R @ np.array(r)))
        lib = {z: [gemqc.ShellDef(0, True, [e * z, 0.4 * e * z], [0.5, 0.6]) for e in (5.0, 0.5)]
               + [gemqc.ShellDef(1, True, [1.1 * z / 9], [1.0])] for z in (1, 9)}
        lib[9] += [gemqc.ShellDef(5, True, [0.9], [1.0]), gemqc.ShellDef(6, True, [1.3], [1.0])]
        wf = gemqc.rhf(mol, gemqc.BasisSet(mol, lib), gemqc.ScfOptions())
        assert wf.converged
        energies.append(wf.energy)
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)
