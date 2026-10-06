"""Basis sets per element and atom (basislib.assign_basis, canonical names, input keys
basis and *_auxbasis, restart files, symmetry lowering)."""

from pathlib import Path

import numpy as np
import pytest

import gemqc
from gemqc import _core
from gemqc.basislib import (aux_assignment, assign_basis, atom_shells, basis_names,
                            build_basis, canonical_name, core_fallback, load_basis,
                            source_name)
from test_qn import quiet  # noqa: F401 (fixture)

DATA = Path(__file__).parent / "data"
GEOM = ("      O1    0.0000000    0.0000000    0.3935881\n"
        "      H     0.7571222    0.0000000   -0.1967941\n"
        "      H_b  -0.7571222    0.0000000   -0.1967941")
ZS = [8, 1, 1]
LABELS = ["O1", "H", "H_b"]


def test_assignment_and_canonical_name():
    """Precedence atom > element > default; labels, numbers, 'No' (YAML false); the
    canonical name reproduces the assignment."""
    a = assign_basis({"default": "cc-pVDZ", "H": "cc-pVTZ", 3: "aug-cc-pVDZ"}, ZS, LABELS)
    assert a == ["cc-pVDZ", "cc-pVTZ", "aug-cc-pVDZ"]
    assert assign_basis({"O1": "x", "h": "y", "H_b": "z"}, ZS, LABELS) == ["x", "y", "z"]
    assert assign_basis("cc-pVDZ", ZS) == ["cc-pVDZ"] * 3
    assert assign_basis({False: "b", "default": "a"}, [102, 1]) == ["b", "a"]
    assert canonical_name(["a", "a", "a"], ZS) == "a"
    assert canonical_name(["a", "b", "b"], ZS) == "b;O=a"
    assert canonical_name(["a", "b", "c", "b", "a"], [8, 1, 1, 1, 6]) == "a;H=b;3=c"  # tie: first
    for names in (["a", "b", "c"], ["a", "a", "b"], ["c", "b", "b"]):
        assert basis_names(canonical_name(names, ZS), ZS) == names
    assert assign_basis("a; H = b; 3=c", ZS) == ["a", "b", "c"]
    assert assign_basis({"H": "b"}, ZS, partial=True) == [None, "b", "b"]


@pytest.mark.parametrize("spec, match", [
    ({"H": "x"}, "no basis set for atom"),
    ({"default": "x", "C": "y"}, "no C atom"),
    ({"default": "x", "Q7": "y"}, "no element symbol, atom number"),
    ({"default": "x", 4: "y"}, "no element symbol, atom number"),
    ({"default": "x", 3: "y", "H_b": "z"}, "named twice"),
    ({"default": "x", "h": "y", "H": "z"}, "duplicate key"),
    ({"default": ["x"]}, "expected a basis set name"),
    ("x;H", "expected 'key=name'"),
])
def test_assignment_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        assign_basis(spec, ZS, LABELS)


def test_core_valence_fallback():
    """H and He of p(w)CVXZ sets (all types) are taken from pVXZ if the file lacks them."""
    assert core_fallback("aug-cc-pwCVTZ-C", 1) == "aug-cc-pVTZ-C"
    assert core_fallback("cc-pCVQZ", 2) == "cc-pVQZ"
    assert core_fallback("aug-cc-pwCVTZ", 8) is None
    assert core_fallback("cc-pVTZ", 1) is None
    for name, alt in (("aug-cc-pwCVTZ", "aug-cc-pVTZ"), ("aug-cc-pwCVTZ-C", "aug-cc-pVTZ-C")):
        assert source_name(name, 1) == alt and source_name(name, 8) == name
        h, h_ref = load_basis(name)[1], load_basis(alt)[1]
        assert [(s.l, s.exponents) for s in h] == [(s.l, s.exponents) for s in h_ref]
    mol = _core.Molecule()
    for z, x in zip(ZS, ([0, 0, 0.74], [1.43, 0, -0.37], [-1.43, 0, -0.37])):
        mol.add_atom(z, x)
    assert build_basis(mol, "aug-cc-pwCVTZ").nbf == 59 + 2 * 23  # O [7s6p4d2f], H [4s3p2d]
    assert aux_assignment("jk", "aug-cc-pwCVTZ", ZS)[0] == "aug-cc-pVTZ-JK"
    assert aux_assignment("jk", "cc-pwCVDZ", ZS)[0] == "cc-pVTZ-JK"
    assert aux_assignment("c", "aug-cc-pwCVTZ", ZS)[0] == "aug-cc-pwCVTZ-C"


@pytest.mark.parametrize("x, nx", [("D", "T"), ("T", "Q"), ("Q", "5")])
def test_f12_auxiliary_defaults(x, nx):
    """cc-p(C)VXZ-F12: JK cc-pV(X+1)Z-JK, C aug-cc-pwCVXZ-C (cc-pCVXZ-F12: own file, H and
    He from aug-cc-pwCVXZ-C), CABS from the files; all atoms of water covered."""
    zs = [8, 1, 1]
    for core in ("", "C"):
        name = f"cc-p{core}V{x}Z-F12"
        expect = {"jk": f"cc-pV{nx}Z-JK", "cabs": f"{name}-CABS",
                  "c": f"aug-cc-pwCV{x}Z-C" + (f";O={name}-C" if core else "")}
        for kind, ref in expect.items():
            aux = aux_assignment(kind, name, zs)[0]
            assert aux == ref
            for z, b in zip(zs, basis_names(aux, zs)):
                assert z in load_basis(b)


def test_atom_basis_set():
    """Per-atom BasisSet equals the per-element one and a hand-built shell list."""
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    plain = build_basis(mol, "cc-pVDZ")
    per_el = build_basis(mol, {"O": "cc-pVDZ", "H": "cc-pVDZ"})
    assert per_el.name == "cc-pVDZ" and per_el.nbf == plain.nbf
    mixed = build_basis(mol, {"default": "cc-pVDZ", 2: "aug-cc-pVDZ"})
    assert mixed.name == "cc-pVDZ;2=aug-cc-pVDZ"
    ref = _core.BasisSet(mol, [load_basis("cc-pVDZ")[8], load_basis("aug-cc-pVDZ")[1],
                               load_basis("cc-pVDZ")[1]], "ref")
    assert mixed.nbf == ref.nbf == 14 + 9 + 5
    assert np.abs(np.asarray(_core.overlap(mixed)) - np.asarray(_core.overlap(ref))).max() == 0
    assert [len(s) for s in atom_shells(mixed.name, ZS)] == [6, 5, 3]
    with pytest.raises(ValueError, match="shells for 2 atoms"):
        _core.BasisSet(mol, [load_basis("cc-pVDZ")[8]] * 2)


def _run(tmp_path, basis, tasks, sym="auto", name="job", coords=None):
    inp = tmp_path / f"{name}.yaml"
    where = coords or "geometry: |\n" + GEOM
    inp.write_text(f"systems:\n  w:\n    symmetry: {sym}\n    basis: {basis}\n    {where}\n"
                   "tasks:\n" + tasks)
    return [r for _, r in gemqc.run(inp, quiet=True)]


def _stored_basis(h5):
    import h5py
    with h5py.File(h5) as f:
        return f["w/scf/basis"].attrs["name"]


def test_scf_and_symmetry(tmp_path):
    """RHF with O in aug-cc-pVDZ (per element, C2v) equals the hand-built basis; one H
    in another basis lowers C2v to Cs (C1 energy, the hand-built basis), an explicit
    c2v is refused."""
    res = _run(tmp_path, "{default: cc-pVDZ, O: aug-cc-pVDZ}",
               "  - {system: w, task: rhf, convergence: 1e-9}\n")
    assert res[0]["energy"] < -76.0
    mol = _core.Molecule()
    for z, x in zip(ZS, ([0, 0, 0.3935881], [0.7571222, 0, -0.1967941],
                         [-0.7571222, 0, -0.1967941])):
        mol.add_atom(z, [c / 0.529177210903 for c in x])  # bohr
    bs = _core.BasisSet(mol, {8: load_basis("aug-cc-pVDZ")[8], 1: load_basis("cc-pVDZ")[1]})
    o = gemqc.ScfOptions()
    o.convergence = 1e-9
    ref = gemqc.rhf(mol, bs, o, gemqc.sad_guess(mol, bs))
    assert res[0]["energy"] == pytest.approx(ref.energy, abs=1e-8)
    with pytest.raises(gemqc.InputError, match="basis sets per atom do not allow c2v"):
        _run(tmp_path, "{default: cc-pVDZ, H_b: aug-cc-pVDZ}", "  - {system: w, task: rhf}\n",
             sym="c2v", name="e")
    mixed = _run(tmp_path, "{default: cc-pVDZ, H_b: aug-cc-pVDZ}",
                 "  - {system: w, task: rhf, convergence: 1e-9}\n", sym="nosym", name="m")
    assert _stored_basis(tmp_path / "m.h5") == "cc-pVDZ;3=aug-cc-pVDZ"
    lowered = _run(tmp_path, "{default: cc-pVDZ, H_b: aug-cc-pVDZ}",
                   "  - {system: w, task: rhf, convergence: 1e-9}\n", name="cs")
    import json
    sym = json.loads((tmp_path / "cs.json").read_text())["systems"]["w"]["symmetry"]
    assert sym["computational_group"] == "Cs" and sym["nuclear_point_group"] == "C2v"
    assert lowered[0]["energy"] == pytest.approx(mixed[0]["energy"], abs=1e-9)
    bs2 = _core.BasisSet(mol, [load_basis("cc-pVDZ")[8], load_basis("cc-pVDZ")[1],
                               load_basis("aug-cc-pVDZ")[1]])
    ref2 = gemqc.rhf(mol, bs2, o, gemqc.sad_guess(mol, bs2))
    assert mixed[0]["energy"] == pytest.approx(ref2.energy, abs=1e-8)


def test_restart_across_bases(tmp_path):
    """A mixed basis is stored per atom; a restart into another basis (mixed -> plain,
    plain -> mixed) converges to the same energy as a fresh start."""
    tasks = "  - {system: w, task: rhf, convergence: 1e-9}\n"
    mixed = "{default: cc-pVDZ, 3: aug-cc-pVDZ}"
    fresh_m = _run(tmp_path, mixed, tasks, sym="nosym", name="fm")[0]["energy"]
    fresh_p = _run(tmp_path, "cc-pVDZ", tasks, sym="nosym", name="fp")[0]["energy"]
    _run(tmp_path, mixed, tasks, sym="nosym", name="r")
    import h5py
    with h5py.File(tmp_path / "r.h5") as f:
        assert list(f["w/scf/basis/atom_basis"][()]) == [0, 0, 1]
    assert _stored_basis(tmp_path / "r.h5") == "cc-pVDZ;3=aug-cc-pVDZ"
    assert _run(tmp_path, "cc-pVDZ", tasks, sym="nosym", name="r")[0]["energy"] == \
        pytest.approx(fresh_p, abs=1e-8)
    assert _run(tmp_path, mixed, tasks, sym="nosym", name="r")[0]["energy"] == \
        pytest.approx(fresh_m, abs=1e-8)


def test_auxiliary_bases(tmp_path):
    """Fitting bases per atom: defaults from the orbital basis of each atom, partial
    system keys and task options on top."""
    zs = [8, 1, 1]
    assert aux_assignment("jk", "cc-pVDZ;O=aug-cc-pVDZ", zs) == (
        "cc-pVTZ-JK;O=aug-cc-pVTZ-JK", "per atom: default table")
    assert aux_assignment("jk", "cc-pVDZ", zs, None, [(None, "t"), ("x", "s")]) == ("x", "s")
    spec = [({"H": "y"}, "task"), ({"O": "x"}, "input")]
    assert aux_assignment("jk", "cc-pVDZ", zs, None, spec) == ("y;O=x", "per atom: input, task")
    assert aux_assignment("jk", "cc-pVDZ", zs, None, [({"O": "x"}, "t")])[0] == \
        "cc-pVTZ-JK;O=x"
    res = _run(tmp_path, "{default: cc-pVTZ, O: aug-cc-pVTZ}",
               "  - {system: w, task: df-rhf}\n  - {task: df-mp2}\n"
               "  - {task: df-mp2, c_auxbasis: {O: cc-pVQZ-C}}\n")
    scf_, mp2, mp2b = res
    assert scf_["density_fitting"]["jk_auxbasis"] == "cc-pVTZ-JK;O=aug-cc-pVTZ-JK"
    assert mp2["density_fitting"]["c_auxbasis"] == "cc-pVTZ-C;O=aug-cc-pVTZ-C"
    assert mp2b["density_fitting"]["c_auxbasis"] == "cc-pVTZ-C;O=cc-pVQZ-C"
    assert mp2b["correlation_energy"] == pytest.approx(mp2["correlation_energy"], abs=1e-3)
    assert mp2b["correlation_energy"] != pytest.approx(mp2["correlation_energy"], abs=1e-7)
    f12 = _run(tmp_path, "{default: cc-pVDZ, O: aug-cc-pVDZ}",
               "  - {system: w, task: df-rhf}\n"
               "  - {task: df-mp2-f12, cabs_auxbasis: {default: aug-cc-pVDZ-CABS},"
               " c_auxbasis: {default: aug-cc-pVDZ-C}}\n", name="f12")[1]
    assert f12["cabs_auxbasis"] == "aug-cc-pVDZ-CABS" and f12["e_f12"] < 0.0


def test_labels_from_files(tmp_path):
    """Atom labels of an XYZ file select atoms; stored geometries have none."""
    (tmp_path / "w.xyz").write_text("3\n\n" + "\n".join(l.strip() for l in GEOM.split("\n"))
                                    + "\n")
    res = _run(tmp_path, "{default: cc-pVDZ, O1: aug-cc-pVDZ}",
               "  - {system: w, task: rhf}\n", coords="coordinates: w.xyz")
    assert _stored_basis(tmp_path / "job.h5") == "cc-pVDZ;O=aug-cc-pVDZ"
    with pytest.raises(gemqc.InputError, match="atom label"):
        _run(tmp_path, "{default: cc-pVDZ, O1: aug-cc-pVDZ}", "  - {system: w, task: rhf}\n",
             coords="coordinates: job.h5", name="b")
