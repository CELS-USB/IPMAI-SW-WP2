IPMAI Cycle Monitor Framework Sphinx Documentation
==================================================

Windows build
-------------

Install Sphinx if needed::

   pip install sphinx

Build the HTML documentation from this folder::

   python -m sphinx -b html . _build\html

Alternatively run::

   build_html.bat

Open after build::

   _build\html\index.html

Notes
-----

The source files use reStructuredText syntax. Do not use Markdown fences such as triple backticks in these Sphinx source files.
