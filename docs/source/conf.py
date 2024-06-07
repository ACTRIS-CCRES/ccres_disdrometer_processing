# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from docs.scripts.plot import make_plots

BASE_URL = "https://github.com/ACTRIS-CCRES/ccres_disdrometer_processing"
# -- Project information -----------------------------------------------------

project = "ccres_disdrometer_processing"
copyright = "ACTRIS-CCRES"
author = "ACTRIS-CCRES"

version_package = {}
with open("../../ccres_disdrometer_processing/__init__.py") as fp:
    exec(fp.read(), version_package)
release = version_package["version_str"]

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets/logo_actris_ccress.png")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "ablog",
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.youtube",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_examples",
    "sphinx_tabs.tabs",
    "sphinx_thebe",
    "sphinx_togglebutton",
    # "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
    # For the kitchen sink
    "sphinx.ext.todo",
]



def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"{BASE_URL}/tree/main/{filename}.py"

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]



autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

language = "en"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_logo = LOGO_PATH
html_favicon = LOGO_PATH
html_title = "CCRES Disdrometer processing"
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": BASE_URL,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "home_page_in_toc": False,
    "show_toc_level": 2,
}

html_sidebars = {
    "reference/blog/*": [
        "navbar-logo.html",
        "search-field.html",
        "ablog/postcard.html",
        "ablog/recentposts.html",
        "ablog/tagcloud.html",
        "ablog/categories.html",
        "ablog/archives.html",
        "sbt-sidebar-nav.html",
    ]
}


html_static_path = ["assets", "_static"]
html_css_files = [
    "css/custom.css",
]

# -- Custom code to run -------------------------------------------------

make_plots()
