"""Orbital analysis: per-orbital Mulliken weights (character), line wrapping."""

from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc.analysis import Character, ao_groups, character_items, mulliken_weights, wrap_items
from gemqc.driver import System
from gemqc.symmetry import symmetrize_molecule
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"


def system(name):
    mol = gemqc.read_xyz(DATA / f"{name}.xyz")
    sym = symmetrize_molecule(mol)
    return System(name, mol, gemqc.build_basis(mol, "def2-SVP"), sym)


def test_groups_merged_and_separate():
    s = system("butane")
    group, names = ao_groups(s.molecule, s.basis, s.symmetry, merge=True)
    assert len(group) == s.basis.nbf
    # gauche butane (C2): two classes of C, five of H
    assert [n for n in names if n.startswith("C")] == ["C(1) s", "C(1) p", "C(1) d",
                                                        "C(2) s", "C(2) p", "C(2) d"]
    assert sum(n.endswith(" s") and n.startswith("H") for n in names) == 5
    _, names = ao_groups(s.molecule, s.basis, s.symmetry, merge=False)
    assert names[:3] == ["1C s", "1C p", "1C d"] and "14H p" in names
    w = system("h2o")
    _, names = ao_groups(w.molecule, w.basis, w.symmetry)
    assert names == ["O s", "O p", "O d", "H s", "H p"]


def test_weights_sum_to_one():
    s = system("h2o")
    wf = gemqc.rhf(s.molecule, s.basis, gemqc.ScfOptions(), gemqc.sad_guess(s.molecule, s.basis))
    ch = Character(s)
    W = mulliken_weights(wf.C, ch.S, ch.group, len(ch.names))
    assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)
    # total Mulliken population of the occupied orbitals: 10 electrons
    assert 2 * W[:, :5].sum() == pytest.approx(10.0)
    items = ch.items(wf.C)
    assert items[0] == ["1.000 O s"]
    assert all(sum(float(t.split()[0]) for t in it) >= 0.6 for it in items)


def test_character_items_and_wrap():
    w = np.array([0.1, -0.2, 0.5, 0.3, 0.0002])
    names = ["a", "b", "c", "d", "e"]
    assert character_items(w, names, 0.6) == ["0.500 c", "0.300 d"]
    assert character_items(w, names, 0.9) == ["0.500 c", "0.300 d", "-0.200 b", "0.100 a"]
    lines = wrap_items("head  ", [f"{i:03d}" for i in range(40)], width=30)
    assert all(len(x) <= 30 for x in lines) and len(lines) > 1
    assert all(x.startswith(" " * 6) and x[6] != " " for x in lines[1:])
