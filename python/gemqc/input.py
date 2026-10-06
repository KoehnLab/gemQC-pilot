"""YAML input parsing.

Input structure (blocks may be written as maps or as lists of one-key maps):

    systems:
      water:
        coordinates: h2o.xyz     # path relative to the input file (.xyz, .h5, .json)
        units: angstrom          # of the coordinates (default: angstrom)
        # instead of coordinates: geometry (inline 'symbol x y z' lines) or
        # zmatrix (inline) with variables: {r: 0.96, a: 104.5}
        basis: def2-SVP
        charge: 0                # default 0
        multiplicity: 1          # default 1
        symmetry: auto           # auto (default), nosym, or D2h subgroup, e.g. c2v
        modify: align            # off, align (default: rotate/translate), adapt
        symmetry_tolerance: 1e-3 # bohr (only with modify: adapt)
    resources:                   # optional
      memory: 4 GB               # limit for main memory (bare number: MB)
      threads: 8                 # default: all cores (OpenMP)
    tasks:
      - system: water            # first task must name a system; later
        task: scf                #   tasks default to the previous one
        convergence: 1e-6        # further keys are task options
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .options import RESOURCES, SYSTEM, InputError, check_keys, resolve

TOP_KEYS = ("systems", "resources", "tasks")


@dataclass
class SystemSpec:
    """A system of the input (defaults: options.SYSTEM)."""
    name: str
    coordinates: Path           # file; None for geometry or zmatrix
    basis: str
    jk_auxbasis: str            # JK fitting basis (None: default table)
    c_auxbasis: str             # correlation fitting basis (None: default table)
    cabs_auxbasis: str          # CABS of F12 methods (None: default table)
    units: str
    charge: int
    multiplicity: int
    symmetry: str               # auto, nosym, or a D2h subgroup (e.g. c2v)
    modify: str                 # off, align, adapt (geometry changes allowed)
    symmetry_tolerance: float   # bohr; None: default of modify
    source_system: str = None   # system in an HDF5/JSON coordinates file
    geometry: str = None        # inline 'symbol x y z' lines
    zmatrix: str = None         # inline Z-matrix
    variables: dict = None      # Z-matrix variables


@dataclass
class TaskSpec:
    """A task of the input: name, system and the options as given."""
    task: str
    system: str
    options: dict = field(default_factory=dict)


@dataclass
class ResourceSpec:
    """Memory limit (bytes, 0: unlimited), threads (0: all cores), scratch directory."""
    memory: int = 0   # bytes, 0: unlimited
    threads: int = 0  # 0: default
    scratch: str = None  # None: $GEMQC_SCRATCH or the temporary directory


@dataclass
class InputSpec:
    """Parsed input: path, systems {name: SystemSpec}, tasks [TaskSpec], resources."""
    path: Path
    systems: dict  # name -> SystemSpec
    tasks: list    # of TaskSpec
    resources: ResourceSpec = field(default_factory=ResourceSpec)


_MEM_UNITS = {"b": 1, "kb": 1024, "k": 1024, "mb": 1024**2, "m": 1024**2,
              "gb": 1024**3, "g": 1024**3, "tb": 1024**4, "t": 1024**4}


def parse_memory(value) -> int:
    """Memory size like '4 GB', '500MB' or 2048 (bare number: MB) in bytes.

    Raises:
        InputError: Invalid size or unit.
    """
    text = str(value).strip().lower().replace(" ", "")
    num = text.rstrip("abcdefghijklmnopqrstuvwxyz")
    unit = text[len(num):] or "mb"
    try:
        return int(float(num) * _MEM_UNITS[unit])
    except (ValueError, KeyError):
        raise InputError(f"invalid memory size '{value}' (e.g. '4 GB', '500 MB')") from None


def _parse_resources(block):
    d = resolve(RESOURCES, _as_mapping(block, "resources"), "resources")
    return ResourceSpec(memory=parse_memory(d["memory"]) if d["memory"] is not None else 0,
                        threads=d["threads"], scratch=d["scratch"])


def _as_mapping(obj, where):
    """Accept a map or a list of maps (merged); None -> {}."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, list):
        merged = {}
        for item in obj:
            if not isinstance(item, dict):
                raise InputError(f"{where}: expected 'key: value' entries, got {item!r}")
            for k, v in item.items():
                if k in merged:
                    raise InputError(f"{where}: duplicate key '{k}'")
                merged[k] = v
        return merged
    raise InputError(f"{where}: expected a block of 'key: value' entries")


def _coerce(value):
    """YAML 1.1 reads e.g. '1e-6' as string; convert numeric-looking strings."""
    if isinstance(value, str):
        for conv in (int, float):
            try:
                return conv(value)
            except ValueError:
                pass
    return value


def _parse_system(name, block, base):
    where = f"system '{name}'"
    block = dict(_as_mapping(block, where))
    for key in ("geometry", "zmatrix"):  # lines may be given as a list
        if isinstance(block.get(key), list):
            block[key] = "\n".join(str(x) for x in block[key])
    d = resolve(SYSTEM, block, where)
    given = [k for k in ("coordinates", "geometry", "zmatrix") if d[k] is not None]
    if len(given) != 1:
        raise InputError(f"{where}: give one of coordinates, geometry, zmatrix"
                         + (f" (not {', '.join(given)})" if given else ""))
    if d["variables"] is not None and d["zmatrix"] is None:
        raise InputError(f"{where}: variables require a zmatrix")
    stored = d["coordinates"] is not None and Path(d["coordinates"]).suffix.lower() in (
        ".h5", ".hdf5", ".json")
    if d["source_system"] is not None and not stored:
        raise InputError(f"{where}: source_system requires an HDF5 or JSON coordinates file")
    if d["symmetry_tolerance"] is not None and d["modify"] != "adapt":
        raise InputError(f"system '{name}': symmetry_tolerance requires modify: adapt "
                         f"(modify: {d['modify']} uses a tight fixed tolerance)")
    if d["coordinates"] is not None:
        d["coordinates"] = (base / d["coordinates"]).resolve()
    return SystemSpec(name=str(name), **d)


def parse_input(path) -> InputSpec:
    """Read and check a YAML input file (systems, resources, tasks).

    Task options are checked later against the task tables (gemqc.options).

    Returns:
        InputSpec.

    Raises:
        InputError: Invalid YAML, unknown keys, missing or invalid values.
    """
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise InputError(f"{path.name}: invalid YAML\n{e}") from None
    top = _as_mapping(raw, path.name)
    check_keys(top, TOP_KEYS, path.name)

    systems = {}
    for name, block in _as_mapping(top.get("systems"), "systems").items():
        systems[str(name)] = _parse_system(name, block, path.parent)

    tasks, current = [], None
    task_list = top.get("tasks") or []
    if not isinstance(task_list, list):
        raise InputError("tasks: expected a list of tasks")
    for n, block in enumerate(task_list, start=1):
        where = f"task {n}"
        d = {k: _coerce(v) for k, v in _as_mapping(block, where).items()}
        if "system" in d:
            current = str(d.pop("system"))
        if current is None:
            raise InputError(f"{where}: no system selected")
        if current not in systems:
            raise InputError(f"{where}: unknown system '{current}'")
        if "task" not in d:
            raise InputError(f"{where}: 'task' is required")
        tasks.append(TaskSpec(task=str(d.pop("task")).lower(), system=current, options=d))
    return InputSpec(path=path, systems=systems, tasks=tasks,
                     resources=_parse_resources(top.get("resources")))
