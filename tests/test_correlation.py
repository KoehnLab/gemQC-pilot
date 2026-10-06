"""Orbital spaces of correlated methods: frozen core, deleted virtuals, semicanonical ROHF."""

import io
from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.correlation import (core_shells, frozen_core_orbitals, orbital_spaces, print_spaces,
                               spin_fock)
from gemqc.options import InputError
from gemqc.output import Output
from gemqc.symmetry import scf_symmetry, symmetrize_molecule
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"


def scf(name, mult=1, method="rhf"):
    mol = gemqc.read_xyz(DATA / f"{name}.xyz")
    mol.multiplicity = mult
    sym = symmetrize_molecule(mol, request="auto")
    bs = gemqc.build_basis(mol, "def2-SVP")
    ss = scf_symmetry(bs, sym) or _core.ScfSymmetry()
    o = gemqc.ScfOptions()
    o.convergence = 1e-8
    guess = gemqc.sad_guess(mol, bs)
    if method == "rhf":
        return gemqc.rhf(mol, bs, o, guess, ss)
    return {"rohf": gemqc.rohf, "uhf": gemqc.uhf}[method](mol, bs, o, [guess], ss)


def test_frozen_core_table():
    ref = {1: 0, 2: 0, 6: 1, 10: 1, 14: 5, 26: 9, 30: 9, 31: 14, 36: 14, 37: 18, 53: 23,
           54: 23, 55: 27, 80: 34, 86: 39, 92: 43}
    assert {z: frozen_core_orbitals(z) for z in ref} == ref


def test_core_shells():
    """Inner and outer core per element (outer: core subshells of the highest n); they
    add up to the frozen core; Cu(NH3)+: 5 inner and 5 outer core orbitals as in Molpro
    (reference_mp2/cu-nh3-*)."""
    ref = {1: (0, 0), 8: (0, 1), 14: (1, 4), 29: (5, 4), 31: (5, 9), 50: (14, 9),
           58: (23, 4), 80: (30, 4), 82: (30, 9), 102: (39, 4)}
    assert {z: core_shells(z) for z in ref} == ref
    assert all(sum(core_shells(z)) == frozen_core_orbitals(z) for z in range(1, 119))


def test_rhf_spaces(quiet):
    wf = scf("h2o")
    sp = orbital_spaces(wf)
    a = sp.alpha
    assert sp.beta is a and sp.reference == "RHF"
    assert (a.nfrozen, a.nocc, a.ndeleted) == (1, 4, 0)
    assert a.per_irrep(a.frozen, 4) == [1, 0, 0, 0]
    assert a.eps[0] == pytest.approx(np.min(wf.eps))
    # the reference orbitals, reordered
    assert np.allclose(np.sort(a.eps), np.sort(np.asarray(wf.eps)))
    sp = orbital_spaces(wf, frozen=[1, 0, 1, 0], deleted=2)
    a = sp.alpha
    assert a.per_irrep(a.frozen, 4) == [1, 0, 1, 0] and a.ndeleted == 2
    assert np.all(a.eps[a.deleted] >= a.eps[a.virt].max())
    assert orbital_spaces(wf, frozen="off").alpha.nfrozen == 0
    assert orbital_spaces(wf, deleted=[1, 0, 0, 0]).alpha.per_irrep(slice(-1, None), 4) == [1, 0, 0, 0]
    buf = io.StringIO()
    print_spaces(sp, list(wf.irrep_names), Output(stream=buf))
    lines = buf.getvalue().splitlines()
    assert any(line.split()[:1] == ["frozen"] and line.split()[1:] == ["1", "0", "1", "0", "2"]
               for line in lines)
    assert not any("core" in line for line in lines)  # O 1s frozen: no correlated core
    # all electrons: O 1s is outer core (energy rank), the rest valence
    a = orbital_spaces(wf, frozen="off").alpha
    assert a.per_irrep(a.occ_of_class(1), 4) == [1, 0, 0, 0]
    assert len(a.occ_of_class(0)) == 0 and len(a.occ_of_class(2)) == 4
    assert a.eps[a.occ_of_class(1)][0] == pytest.approx(np.min(wf.eps))


@pytest.mark.parametrize("mode", ["freeze_first", "diag_first"])
def test_rohf_semicanonical(quiet, mode):
    wf = scf("ch2", 3, "rohf")
    sp = orbital_spaces(wf, frozen=1, deleted=3, semicanonical=mode)
    Fa, Fb = spin_fock(wf)
    S = np.asarray(_core.overlap(wf.basis))
    C, occ = np.asarray(wf.C), np.asarray(wf.occ)
    for s, F, nocc in ((sp.alpha, Fa, 5), (sp.beta, Fb, 3)):
        assert np.allclose(s.C.T @ S @ s.C, np.eye(s.C.shape[1]), atol=1e-10)
        assert np.allclose(s.F, s.C.T @ F @ s.C)
        assert s.nfrozen + s.nocc == nocc and s.nfrozen == 1 and s.ndeleted == 3
        # diagonal within the diagonalized blocks
        blocks = [s.occ, s.virt] if mode == "freeze_first" else [
            slice(0, nocc), slice(nocc, s.C.shape[1])]
        for b in blocks:
            f = s.F[b, b]
            assert np.allclose(f - np.diag(np.diag(f)), 0.0, atol=1e-9)
            assert np.allclose(np.diag(f), s.eps[b])
        # occupied space of this spin unchanged
        Po = s.C[:, :nocc] @ s.C[:, :nocc].T
        Cr = C[:, occ > (0.5 if nocc == 5 else 1.5)]
        assert np.allclose(Po, Cr @ Cr.T, atol=1e-9)
    if mode == "freeze_first":  # frozen and deleted: reference orbitals, same for both spins
        assert np.allclose(sp.alpha.C[:, sp.alpha.frozen], sp.beta.C[:, sp.beta.frozen])
        assert np.allclose(sp.alpha.C[:, sp.alpha.deleted], sp.beta.C[:, sp.beta.deleted])
        assert np.allclose(np.abs(sp.alpha.C[:, 0]), np.abs(C[:, np.argmin(wf.eps)]))


def test_uhf_spaces(quiet):
    wf = scf("ch2", 3, "uhf")
    sp = orbital_spaces(wf, deleted=[1, 0, 0, 0])
    assert sp.reference == "UHF"
    assert (sp.alpha.nfrozen, sp.alpha.nocc, sp.beta.nfrozen, sp.beta.nocc) == (1, 4, 1, 2)
    assert sp.alpha.ndeleted == sp.beta.ndeleted == 1


def test_errors(quiet):
    wf = scf("h2o")
    with pytest.raises(InputError, match="only 5 available"):
        orbital_spaces(wf, frozen=6)
    with pytest.raises(InputError, match="4 numbers expected"):
        orbital_spaces(wf, frozen=[1, 0])
    with pytest.raises(InputError, match="in irrep 2"):
        orbital_spaces(wf, frozen=[0, 9, 0, 0])
    with pytest.raises(InputError, match="freeze_first or diag_first"):
        orbital_spaces(wf, semicanonical="x")
    with pytest.raises(InputError, match="expected auto, off"):
        orbital_spaces(wf, frozen="all")
