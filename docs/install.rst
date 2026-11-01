Installation
============

Quick install
-------------

Install from PyPI::

    pip install networkx-backbone

Full installation
-----------------

For full functionality, install with optional dependencies. NumPy and SciPy
are needed for statistical methods (:func:`~networkx_backbone.marginal_likelihood_filter`,
:func:`~networkx_backbone.ecm_filter`), bipartite methods
(:func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`),
:func:`~networkx_backbone.doubly_stochastic_filter`,
:func:`~networkx_backbone.local_path_index`, and KS measures
(:func:`~networkx_backbone.ks_degree`, :func:`~networkx_backbone.ks_weight`),
and statistically validated hypergraphs
(:func:`~networkx_backbone.statistically_validated_hypergraph`)::

    pip install "networkx-backbone[full]"

The plotting helpers in :mod:`networkx_backbone.visualization` need
Matplotlib::

    pip install "networkx-backbone[viz]"

Calling a function whose optional dependency is missing raises
:class:`ImportError`; everything else keeps working with NetworkX alone.

Hypergraph interoperability
---------------------------

The ``from_*``/``to_*`` adapters in :mod:`networkx_backbone.hypergraph_io`
import their target library lazily, so none of them is a dependency. Install
the ones you use: ``xgi``, ``hypernetx``, ``hypergraphx``, or
``Hypergraph-Analysis-Toolbox``. Reading and writing the HIF interchange format
(:func:`~networkx_backbone.read_hif`, :func:`~networkx_backbone.write_hif`) needs
only the standard library.

Conda installation
------------------

Install from anaconda.org::

    conda install -c brianckeegan networkx-backbone

Development installation
------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/brianckeegan/networkx_backbone.git
    cd networkx_backbone
    pip install -e ".[dev]"

Run the test suite (including docstring examples) to verify the installation::

    pytest --doctest-modules networkx_backbone tests

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Status
   * - Python
     - >= 3.10
     - Required
   * - NetworkX
     - >= 3.0
     - Required
   * - NumPy
     - >= 1.23
     - Optional (for ``[full]``)
   * - SciPy
     - >= 1.9
     - Optional (for ``[full]``)
   * - Matplotlib
     - >= 3.7
     - Optional (for ``[viz]``)
