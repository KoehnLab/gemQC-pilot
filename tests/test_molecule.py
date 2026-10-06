import numpy as np
import pytest

import gemqc
from gemqc.units import BOHR_IN_ANGSTROM

DATA = __import__("pathlib").Path(__file__).parent / "data"


def test_elements():
    assert gemqc.element_z("O") == 8
    assert gemqc.element_z("cl") == 17
    assert gemqc.element_symbol(26) == "Fe"
    with pytest.raises(ValueError):
        gemqc.element_z("Xx")


def test_read_xyz_units():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    assert mol.natoms == 3 and mol.symbols == ["O", "H", "H"]
    assert mol.coordinates[1, 0] == pytest.approx(0.7571222 / BOHR_IN_ANGSTROM)
    mol_b = gemqc.read_xyz(DATA / "h2o.xyz", unit="bohr")
    assert mol_b.coordinates[1, 0] == pytest.approx(0.7571222)


def test_nuclear_repulsion_and_electrons():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    xyz, z = mol.coordinates, np.array(mol.atomic_numbers)
    ref = sum(z[i] * z[j] / np.linalg.norm(xyz[i] - xyz[j])
              for i in range(3) for j in range(i))
    assert mol.nuclear_repulsion() == pytest.approx(ref, rel=1e-14)
    assert mol.n_electrons == 10


def test_modify_from_python():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    xyz = mol.coordinates
    xyz[1:] *= 2.0
    mol.coordinates = xyz
    assert mol.coordinates[1, 0] == pytest.approx(xyz[1, 0])
    mol.set_position(0, [0.0, 0.0, 0.0])
    assert np.allclose(mol.coordinates[0], 0.0)


def test_charge_multiplicity_validation():
    mol = gemqc.read_xyz(DATA / "h2o.xyz")
    mol.charge = 1
    mol.multiplicity = 2
    mol.validate()
    mol.multiplicity = 1
    with pytest.raises(ValueError):
        mol.validate()


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    v = b0 - (b0 @ b1) / (b1 @ b1) * b1
    w = b2 - (b2 @ b1) / (b1 @ b1) * b1
    x = v @ w
    y = np.cross(b1 / np.linalg.norm(b1), v) @ w
    return np.degrees(np.arctan2(y, x))


def test_zmatrix():
    from gemqc.geometry import parse_zmatrix
    zmat = "O\nH1 1 r\nH2 O1 r 2 a\n"
    mol = parse_zmatrix(zmat.replace("O1", "1"), {"r": 0.96, "a": 104.5}, name="w")
    X = mol.coordinates * BOHR_IN_ANGSTROM
    assert np.allclose(np.linalg.norm(X[1:] - X[0], axis=1), 0.96)
    u, v = X[1] - X[0], X[2] - X[0]
    assert np.degrees(np.arccos(u @ v / (u @ u))) == pytest.approx(104.5)
    assert np.allclose(X[:, 1], 0) and np.allclose(X[1, :2], 0)  # xz plane, 2nd atom on z
    # H2O2 with a dummy atom and a negative variable; units bohr
    zmat = "O\nO 1 roo\nX 1 1.0 2 90\nH 1 roh 2 a 3 d\nH 2 roh 1 a 4 -t\n"
    mol = parse_zmatrix(zmat, {"roo": 2.8, "roh": 1.8, "a": 100, "d": 30, "t": 115}, "bohr")
    X = mol.coordinates
    assert mol.symbols == ["O", "O", "H", "H"]
    assert np.linalg.norm(X[3] - X[1]) == pytest.approx(1.8)
    assert _dihedral(X[2], X[0], X[1], X[3]) == pytest.approx(-115)
    for bad, msg in (("O\nH 1\n", "expected 3"), ("O\nH 2 1.0\n", "unknown reference"),
                     ("O\nH 1 r\n", "unknown variable")):
        with pytest.raises(ValueError, match=msg):
            parse_zmatrix(bad)


def test_geometry_sources_in_input(tmp_path):
    import json
    xyz = (DATA / "h2o.xyz").read_text().splitlines()[2:]
    ref = gemqc.read_xyz(DATA / "h2o.xyz").coordinates
    (tmp_path / "a.yaml").write_text(
        "systems:\n  w:\n    basis: def2-SVP\n    modify: off\n    geometry: |\n"
        + "".join(f"      {l.strip()}\n" for l in xyz)
        + "  z:\n    basis: def2-SVP\n    zmatrix: [O, H 1 r, H 1 r 2 a]\n"
          "    variables: {r: 0.96, a: 104.5}\n"
          "tasks:\n  - {task: oneint, system: w}\n  - {task: oneint, system: z}\n")
    gemqc.run(tmp_path / "a.yaml", quiet=True)
    stored = json.loads((tmp_path / "a.json").read_text())["systems"]
    assert np.allclose(stored["w"]["coordinates_bohr"], ref)
    # geometry from the JSON file of the earlier run; source_system selects
    (tmp_path / "b.yaml").write_text(
        "systems:\n  w: {basis: def2-SVP, coordinates: a.json, source_system: z}\n"
        "tasks:\n  - {task: oneint, system: w}\n")
    gemqc.run(tmp_path / "b.yaml", quiet=True)
    b = json.loads((tmp_path / "b.json").read_text())["systems"]["w"]
    assert np.allclose(b["coordinates_bohr"], stored["z"]["coordinates_bohr"], atol=1e-12)
    for text, msg in (("coordinates: a.json", "select one of the systems"),
                      ("coordinates: a.json, source_system: q", "no system 'q'"),
                      ("coordinates: x.xyz, zmatrix: O", "give one of"),
                      ("geometry: O 0 0 0, variables: {r: 1}", "require a zmatrix"),
                      ("coordinates: x.xyz, source_system: w", "requires an HDF5")):
        (tmp_path / "c.yaml").write_text(
            f"systems:\n  w: {{basis: def2-SVP, {text}}}\ntasks:\n  - {{task: oneint, system: w}}\n")
        with pytest.raises(gemqc.InputError, match=msg):
            gemqc.run(tmp_path / "c.yaml", quiet=True)


def test_geometry_from_hdf5(tmp_path):
    from gemqc.geometry import read_stored
    (tmp_path / "h2o.xyz").write_text((DATA / "h2o.xyz").read_text())
    (tmp_path / "a.yaml").write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                                     "tasks:\n  - {task: scf, system: w}\n")
    gemqc.run(tmp_path / "a.yaml", quiet=True)
    mol, name = read_stored(tmp_path / "a.h5")
    assert name == "w" and mol.symbols == ["O", "H", "H"]
    ref = gemqc.read_xyz(DATA / "h2o.xyz").coordinates
    dist = lambda X: np.linalg.norm(X[:, None] - X[None], axis=2)
    assert np.allclose(dist(mol.coordinates), dist(ref), atol=1e-10)
