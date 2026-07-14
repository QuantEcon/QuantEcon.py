Contribute to QuantEcon.py
==========================

If you would like to contribute to `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`_,
a good place to start is the `project issue tracker <https://github.com/QuantEcon/QuantEcon.py/issues>`_.

Set up a development environment
--------------------------------

We recommend developing QuantEcon.py inside an isolated environment, so that you can work against your
development version of the package without disturbing the Python environment your other work depends on.

The repository ships a conda ``environment.yml`` (named ``qe``) that contains the scientific stack along
with the development tools (``pytest``, ``flake8`` and ``flit``). To clone the repository and create and
activate the environment:

.. code:: bash

    git clone https://github.com/QuantEcon/QuantEcon.py
    cd QuantEcon.py
    conda env create -f environment.yml
    conda activate qe

QuantEcon.py uses `flit <https://flit.pypa.io>`_ as its build backend. Install your development copy in
editable mode so that changes to the source are picked up immediately:

.. code:: bash

    flit install --symlink

You can learn more about `managing conda environments here
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Write tests
-----------

All functions and methods contributed to QuantEcon.py should be paired with tests to verify that they
are functioning correctly.

Run the test suite with `pytest <https://docs.pytest.org>`_:

.. code:: bash

    pytest quantecon/

We also check code style with `flake8 <https://flake8.pycqa.org>`_. To run the same checks as
continuous integration:

.. code:: bash

    flake8 --select=F401,F405,E231 quantecon

Write documentation
--------------------

We try to maintain a simple and consistent format for inline documentation, known in the Python world as
docstrings.

The format we use is known as `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

It was developed by the numpy and scipy teams and is used in many popular packages.

Adhering to this standard helps us

*   Provide a sense of consistency throughout the library
*   Give users instant access to necessary information at the interpreter prompt (either via the built-in
    Python function ``help(object_name)`` or the Jupyter ``object_name?``)
*   Easily generate a reference manual using sphinx's autodoc and apidoc

It is always useful to build the docs locally before opening a pull request, so that you can check how
your docstrings render in HTML. The documentation is built with `Sphinx <https://www.sphinx-doc.org>`_:

.. code:: bash

    pip install -r docs/rtd-requirements.txt
    cd docs
    make html

The rendered pages are written to ``docs/build/html``. Once you open a pull request, a preview of the
documentation is also built automatically by `Read the Docs <https://readthedocs.org>`_ and linked from
the pull request checks.

Further questions
-----------------

We encourage you to reach out to the `QuantEcon team <https://quantecon.org/team>`_ on the
`Discourse forum <https://discourse.quantecon.org/>`_ if you have any further questions.
