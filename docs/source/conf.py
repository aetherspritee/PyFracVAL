import os
import sys

from pyfracval import __version__, _authors

# Adjust the path to go up two levels from docs/source/ to the project root
sys.path.insert(0, os.path.abspath("../../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = name = "PyFracVAL"
author = _authors
copyright = f"2025, {_authors}"
version, release = __version__, __version__.split("+")[0]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Core Sphinx library to pull documentation from docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' documentation (e.g. Python, NumPy)
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx_autodoc_typehints",  # Automatically document types based on type hints
    "autoapi.extension",
    "myst_parser",  # Parse Markdown files
    "sphinx_copybutton",  # Add copy buttons to code blocks
    "sphinxcontrib.bibtex",  # For BibTeX citation support
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",  # Enables ::: directives for admonitions, etc.
    "deflist",  # Enables definition lists
    "linkify",  # Auto-detect URLs and make them links (use with caution)
    "tasklist",  # Enable checklists - [ ] / - [x]
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
# html_theme = "furo"
html_theme = "pydata_sphinx_theme"
# html_logo = "_static/logo.png" # Optional: Add a logo file to _static/
# html_favicon = "_static/favicon.ico" # Optional: Add a favicon
html_static_path = ["_static"]

# Napoleon config
# napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False  # Include __init__ docstrings
napoleon_include_private_with_doc = False  # Usually False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
napoleon_preprocess_types = False  # Let sphinx-autodoc-typehints handle types
napoleon_type_aliases = None
napoleon_attr_annotations = True

# autodoc
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
inheritance_alias = {}

# autoapi
autoapi_dirs = ["../../pyfracval"]
autoapi_options = [
    "members",
    # "undoc-members",
    # "private-members",
    # "show-inheritance",
    # "show-module-summary",
    # "special-members",  # '__init__' etc.
    "imported-members",
]
autoapi_ignore = ["*migrations*", "__init__*"]
autoapi_add_toctree_entry = True

## Use sphinx-autodoc-typehints setup
# Add parameter types from Napoleon processing
always_document_param_types = True
# Show short names for types (e.g. ndarray instead of numpy.ndarray)
typehints_fully_qualified = False
# Process return type hints
typehints_document_rtype = True
# Don't use napoleon rtype processing, let extension handle it
# typehints_use_rtype = False
# Show default values after comma, 'braces' is other option
# typehints_defaults = "comma"
# Optional: Simplify representation of complex types like Union[str, Path]
# typehints_formatter = lambda annotation, config: repr(annotation)
always_use_bars_union = True

## BibTeX Configuration: Tell the extension where your .bib file is:
bibtex_bibfiles = ["references.bib"]  # Assumes references.bib is in docs/source/
bibtex_default_style = "unsrt"  # Common numeric style, others: plain, alpha
# Optional: Control how citations look, e.g. [(Morán et al. 2019)]
bibtex_reference_style = "author_year"

## Intersphinx Configuration: Set up links to external documentation:
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    # Add others like scipy, pandas if you use/reference them
}
