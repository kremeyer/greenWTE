"""Configuration file for the Sphinx documentation builder."""

import os

os.environ.setdefault("GREENWTE_ENV", "CPU")  # no GPU available for doc builds

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenWTE import __version__

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "greenWTE"
copyright = "2025, Laurenz Kremeyer"
author = "Laurenz Kremeyer"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinxarg.ext",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# doctest settings
doctest_global_setup = """
import greenWTE.tests  # download test assets
from greenWTE.tests.defaults import SI_INPUT_PATH
"""

# matplotlib settings
plot_include_source = True
plot_formats = [("png", 300)]
plot_rcparams = {"figure.figsize": (5, 3.75)}
plot_html_show_source = True
plot_pre_code = doctest_global_setup

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]


# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_private_with_doc = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "special-members": "__init__,__getitem__,__iter__",
}
