"""Scratch files for out-of-core data (e.g. MP2 and F12 integrals).

The directory is `resources: scratch` of the input, else $GEMQC_SCRATCH, else the
system temporary directory; it should be on a fast local disk (SSD).
"""

import os
import tempfile
from pathlib import Path

_directory = None


def set_directory(path):
    """Scratch directory for this run (None: default)."""
    global _directory
    _directory = Path(path) if path else None


def directory() -> Path:
    """Scratch directory: set by the input, else $GEMQC_SCRATCH, else the temp. dir."""
    d = _directory or os.environ.get("GEMQC_SCRATCH") or tempfile.gettempdir()
    return Path(d)


def new_file(prefix, suffix=".h5") -> Path:
    """Path of a new (empty, existing) scratch file; the caller deletes it."""
    d = directory()
    d.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=d)
    os.close(fd)
    return Path(name)
