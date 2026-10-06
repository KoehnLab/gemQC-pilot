"""Sphinx configuration (MyST Markdown). Build: sphinx-build -b html docs docs/_build/html"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_ext"))

project = "gemQC"
author = "A. Koehn"
copyright = "2026, A. Koehn, University of Stuttgart"
try:
    from gemqc import __version__ as release
except ImportError:  # docs without an installed package (no API reference)
    release = "unknown"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "gemqc_options",            # option tables generated from gemqc.options
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True     # dataclass attributes documented once
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}

exclude_patterns = ["_build", "_generated"]  # tables are included, not pages
try:
    import furo  # noqa: F401
    html_theme = "furo"
except ImportError:
    html_theme = "alabaster"
html_title = f"gemQC {release}"
