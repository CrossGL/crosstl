"""Sphinx configuration for CrossGL Translator documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "CrossGL Translator"
author = "CrossGL"
copyright = "2026, CrossGL"
release = "3.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

try:
    import breathe  # noqa: F401
except Exception:
    breathe = None
else:
    extensions.append("breathe")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = ["gast", "jinja2", "numpy"]
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

breathe_projects = {
    "crossgl-translator": str(ROOT / "docs" / "doxygen" / "build" / "xml"),
}
breathe_default_project = "crossgl-translator"

html_theme = "furo"
html_title = "CrossGL Translator"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
