# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

# Ensure local package imports resolve during docs builds.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "networkx-backbone"
copyright = "2025-2026, Brian C. Keegan"
author = "Brian C. Keegan"
# Single source of truth: the version setuptools-scm derives from git tags.
try:
    release = _pkg_version("networkx-backbone")
except PackageNotFoundError:  # building docs from an uninstalled checkout
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon / numpydoc settings --------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
numpydoc_show_class_members = False
numpydoc_validation_checks = set()
numpydoc_xref_param_type = False

# -- Nitpick settings --------------------------------------------------------
# Suppress warnings for type references that numpydoc extracts from docstrings
# (e.g. "graph", "string", "optional", "default=...") which are not real
# Python classes.

nitpick_ignore_regex = [
    # Common type names from NumPy-style docstrings that are not real classes
    (r"py:class", r"graph"),
    (r"py:class", r"string"),
    (r"py:class", r"optional"),
    (r"py:class", r"default.*"),
    (r"py:class", r"int"),
    (r"py:class", r"float"),
    (r"py:class", r"bool"),
    (r"py:class", r"dict"),
    (r"py:class", r"list"),
    (r"py:class", r"callable"),
    (r"py:class", r"iterable"),
    (r"py:class", r"integer"),
    (r"py:class", r"array-like"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"random_state"),
    (r"py:class", r"matplotlib Axes"),
    (r"py:class", r"color"),
    (r"py:class", r"fig"),
    (r"py:class", r"ax"),
    (r"py:class", r"diff"),
    (r"py:class", r"networkx\.Graph"),
    (r"py:class", r"networkx\.DiGraph"),
    (r"py:class", r"or"),
    # Quoted string values and set-like type annotations from docstrings
    (r"py:class", r'".*"'),
    (r"py:class", r'\{.*'),
    (r"py:class", r'.*\}'),
    # NetworkX exceptions named without the module in numpydoc Raises sections
    (r"py:exc", r"NetworkXError"),
    (r"py:exc", r"NetworkXNotImplemented"),
    (r"py:exc", r"AmbiguousSolution"),
]

suppress_warnings = ["ref.citation", "ref.class"]

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Sphinx Gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "backreferences_dir": "auto_examples/backreferences",
    "doc_module": ("networkx_backbone",),
    "reference_url": {"networkx_backbone": None},
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/brianckeegan/networkx_backbone",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "brianckeegan",
    "github_repo": "networkx_backbone",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path = ["_static"]
