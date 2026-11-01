Contributing
============

Contributions to ``networkx-backbone`` are welcome.

Development setup
-----------------

Clone the repository and install in editable mode with all development
dependencies::

    git clone https://github.com/brianckeegan/networkx_backbone.git
    cd networkx_backbone
    pip install -e ".[dev,docs]"

Running tests
-------------

Run the test suite and the docstring examples with pytest::

    pytest --doctest-modules networkx_backbone tests

Measure coverage::

    pytest --cov=networkx_backbone --cov-report=term-missing

``tests/test_minimal_deps.py`` runs the pure-Python methods in a subprocess in
which numpy, scipy, and matplotlib cannot be imported. If you add a method that
needs an optional dependency, import it inside the function (not at module
level) so ``import networkx_backbone`` keeps working with NetworkX alone.

Continuous integration (``.github/workflows/tests.yml``) runs lint, the test
suite on Python 3.10-3.14, and the minimal-dependency smoke tests on every push
and pull request.

Building documentation
-----------------------

Build the documentation locally::

    cd docs
    make html

The built documentation will be in ``docs/_build/html/``.

Code style
----------

The project follows the conventions of NetworkX and the wider scientific-Python
stack:

- Lint and format with `ruff <https://docs.astral.sh/ruff/>`_ (configured in
  ``pyproject.toml``)::

      ruff check networkx_backbone tests
      ruff format networkx_backbone tests

- Take a graph ``G`` as the first argument, return a new graph, and never
  mutate the input.
- Name edge-weight parameters ``weight`` and random-state parameters ``seed``;
  accept ``None``, an ``int``, or a NumPy ``Generator``/``RandomState``, as
  NetworkX does.
- Raise :class:`networkx.NetworkXError` for invalid graph input and
  :class:`networkx.NetworkXNotImplemented` (via
  :func:`networkx.utils.not_implemented_for`) for unsupported graph types;
  raise :class:`ValueError` for invalid non-graph arguments.
- Name score attributes ``<method>_pvalue``, ``<method>_score``, or
  ``<method>_keep`` so they compose with the filters in
  :mod:`networkx_backbone.filters`.
- Use NumPy-style docstrings for all public functions
- Include ``Parameters``, ``Returns``, ``Raises`` (if applicable),
  ``References``, and ``Examples`` sections in docstrings
- Use RST ``.. math::`` directives for mathematical formulas
- Use ``:func:`` and ``:mod:`` roles for cross-references within docstrings


Publishing to PyPI
------------------

PyPI releases are published automatically from the
``.github/workflows/publish-to-PyPI.yml`` workflow when a GitHub Release is
published. The workflow follows the PyPA trusted publishing guide:
https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

Package versions are derived automatically from Git tags using
``setuptools-scm``. Do not edit a version string in ``pyproject.toml``.

Maintainers should configure PyPI trusted publishing once in the PyPI project
settings:

1. Open the ``networkx-backbone`` project on PyPI and add a trusted publisher
   for this GitHub repository.
2. Use the workflow filename ``publish-to-PyPI.yml`` and environment name
   ``pypi``.
3. Create and push a version tag (for example ``v0.2.0``) and then publish a
   GitHub Release from that tag; the workflow will run tests, build
   distributions, verify them with ``twine check``, and publish to PyPI using
   that tag as the package version.

Submitting changes
------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite to verify
5. Submit a pull request
