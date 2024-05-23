# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# sys.path.insert('/usr/lib/python3.9/lib-dynload')
# sys.path.insert('/usr/lib/python39.zip')
sys.path.insert(0, os.path.abspath('../../'))

print(sys.path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'maui-software'
copyright = '2023, Caio Ferreira Bernardo'
author = 'Caio Ferreira Bernardo'
release = '0.1.24'
version = '0.1.24'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'sphinx.ext.githubpages',
#    'sphinx.ext.napoleon',
    'numpydoc',  # docstring examples
    'sphinx.ext.autosectionlabel',
    # 'sphinx_gallery.gen_gallery',
]
# html4_writer = True
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ['matplotlib']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = '../../maui/data/logo/color_logo_no_background.svg'
html_favicon = '../../maui/data/logo/favicon.ico'
#html_theme_options = {'logo_only': True}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If false, no module index is generated.
html_domain_indices = True
# If false, no index is generated.
html_use_index = True
html_use_modindex = True
