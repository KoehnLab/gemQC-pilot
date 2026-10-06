import io
import json
from pathlib import Path

import pytest

import gemqc
from gemqc.basislib import contraction_pattern, find_basis_file, load_basis
from gemqc.driver import print_distances
from gemqc.input import InputError, parse_input
from gemqc.output import Output

LIST_STYLE = """
systems:
  - water:
    - coordinates: data/h2o.xyz
    - basis: def2-SVP
tasks:
  - - system: water
    - task: scf
    - convergence: 1e-6
  - - task: localise
"""

MAP_STYLE = """
systems:
  water:
    coordinates: data/h2o.xyz
    basis: def2-svp
    charge: 1
    multiplicity: 2
tasks:
  - {system: water, task: oneint}
"""


def write(tmp_path, text):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "h2o.xyz").write_text(
        (__import__("pathlib").Path(__file__).parent / "data" / "h2o.xyz").read_text())
    f = tmp_path / "in.yaml"
    f.write_text(text)
    return f


def test_list_style(tmp_path):
    spec = parse_input(write(tmp_path, LIST_STYLE))
    sys_ = spec.systems["water"]
    assert sys_.coordinates == (tmp_path / "data" / "h2o.xyz").resolve()
    assert [t.task for t in spec.tasks] == ["scf", "localise"]
    assert spec.tasks[1].system == "water"            # inherited
    assert spec.tasks[0].options == {"convergence": 1e-6}  # '1e-6' coerced


def test_map_style_and_run(tmp_path):
    f = write(tmp_path, MAP_STYLE)
    spec = parse_input(f)
    assert spec.systems["water"].charge == 1
    (task, res), = gemqc.run(f, quiet=True)
    assert res["S"].shape == (24, 24)


@pytest.mark.parametrize("text, msg", [
    (LIST_STYLE.replace("coordinates", "coordintes"), "did you mean 'coordinates'"),
    (LIST_STYLE.replace("- system: water", "- system: ice"), "unknown system"),
    (LIST_STYLE.replace("- system: water\n    - task", "- task"), "no system selected"),
    (MAP_STYLE.replace("multiplicity: 2", "multiplicity: 1"), None),
    (MAP_STYLE.replace("charge: 1", "symmetry_tolerance: 1e-4"), "requires modify: adapt"),
    (MAP_STYLE.replace("charge: 1", "modify: symmetrize"), "invalid value for 'modify'"),
    (MAP_STYLE.replace("charge: 1", "modify: off\n    symmetry_tolerance: 1e-3"), "modify: off"),
])
def test_input_errors(tmp_path, text, msg):
    f = write(tmp_path, text)
    with pytest.raises(InputError, match=msg):
        gemqc.run(f, quiet=True)


def test_basis_lookup(monkeypatch, tmp_path):
    assert find_basis_file("DEF2-svp").name == "def2-SVP.json"
    with pytest.raises(FileNotFoundError):
        find_basis_file("no-such-basis")
    # $GEMQC_DIR takes precedence
    (tmp_path / "lib" / "basis").mkdir(parents=True)
    (tmp_path / "lib" / "basis" / "def2-SVP.json").write_text("{}")
    monkeypatch.setenv("GEMQC_DIR", str(tmp_path))
    assert find_basis_file("def2-SVP").parent == tmp_path / "lib" / "basis"


def test_contraction_pattern():
    assert contraction_pattern(load_basis("def2-SVP")[8]) == "(7s4p1d) -> [3s2p1d]"


def test_option_tables():
    """Option tables: defaults filled, values converted and checked; SCF solver
    defaults are those of the C++ ScfOptions."""
    from gemqc.options import SOLVER, TASKS, markdown_table, resolve
    o = resolve(TASKS["scf"].options, {"convergence": "1e-7", "orbitals": "LONG"}, "scf")
    assert o["convergence"] == 1e-7 and o["orbitals"] == "long" and o["save"] is True
    assert o["solver"] == gemqc.ScfOptions().solver and TASKS["rhf"] is TASKS["scf"]
    for bad in ({"max_iterations": 1.5}, {"character": "x"}, {"guess": "huckel"}):
        with pytest.raises(InputError, match="invalid value"):
            resolve(TASKS["scf"].options, bad, "scf")
    with pytest.raises(InputError, match="'gammas' is required"):
        resolve(TASKS["mp2-f12xg-pilot"].options, {}, "mp2-f12xg-pilot")
    assert {o.name for o in SOLVER} <= {k for k in dir(gemqc.ScfOptions()) if k[0] != "_"}
    table = markdown_table(TASKS["oneint"].options)
    assert "| `print_matrices` | bool | false | print the matrices |" in table


@pytest.mark.parametrize("unit, scale, warn", [("angstrom", 1, None), ("bohr", 1, "Very short"),
                                               ("angstrom", 10, "very long")])
def test_distance_check(unit, scale, warn):
    mol = gemqc.read_xyz(Path(__file__).parent / "data" / "h2o.xyz", unit=unit)
    mol.coordinates = scale * mol.coordinates
    s = io.StringIO()
    print_distances(mol, Output(stream=s))
    text = s.getvalue()
    assert "shortest distance" in text and "longest distance" in text
    assert warn in text if warn else "WARNING" not in text


def test_archive_option(tmp_path):
    """-a sets path and base name of the JSON and HDF5 files (directories created)."""
    from gemqc.__main__ import main
    (tmp_path / "h2o.xyz").write_text((Path(__file__).parent / "data" / "h2o.xyz").read_text())
    inp = tmp_path / "w.yaml"
    inp.write_text("systems:\n  w: {coordinates: h2o.xyz, basis: def2-SVP}\n"
                   "tasks:\n  - {system: w, task: scf}\n")
    main(["-q", "-a", str(tmp_path / "runs" / "first.json"), str(inp)])
    assert (tmp_path / "runs" / "first.json").exists() and (tmp_path / "runs" / "first.h5").exists()
    assert not (tmp_path / "w.json").exists()
    gemqc.run(inp, quiet=True, archive=tmp_path / "runs" / "first")  # restart from it
    data = json.loads((tmp_path / "runs" / "first.json").read_text())
    assert data["tasks"][0]["results"]["iterations"] <= 2
    # dots in the base name are kept (job identifiers); only .json / .h5 are stripped
    main(["-q", "-a", str(tmp_path / "project.296471-20260928"), str(inp)])
    main(["-q", "-a", str(tmp_path / "project.296472-20260928.h5"), str(inp)])
    for job in ("296471", "296472"):
        for ext in ("json", "h5"):
            assert (tmp_path / f"project.{job}-20260928.{ext}").exists()
    dotted = tmp_path / "w.v2.yaml"  # default: base name of the input without .yaml
    dotted.write_text(inp.read_text())
    gemqc.run(dotted, quiet=True)
    assert (tmp_path / "w.v2.json").exists() and (tmp_path / "w.v2.h5").exists()
