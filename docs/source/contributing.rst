Contribute to QuantEcon.py
==========================

If you would like to contribute to `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`_, 
a good place to start is the `project issue tracker <http://github.com/QuantEcon/QuantEcon.py/issues>`_. 

Set up a Conda development environment
--------------------------------------

One of the advantages of the `Anaconda Python environment <https://www.anaconda.com/distribution/>`_ is that it is
cheap to set up (and discard) Python environments for development versions of packages and populate them with your
favorite scientific tools. 

For example, if you're working on QuantEcon.py you might find it useful to set up an
environment (containing NumPy, SciPy, etc.) that uses your development version rather than the default ones. 

This facilitates contributing to QuantEcon.py without worrying about corrupting the Python environment on which your other work depends.

You can learn more about `managing environments here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

Write tests
-----------

All functions and methods contributed to QuantEcon.py should be paired with tests to verify that they are functioning correctly.

Write documentation
-------------------

We try to maintain a simple and consistent format for inline documentation, known in the Python world as docstrings. 

The format we use is known as `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_. 

It was developed by the numpy and scipy teams and is used in many popular packages. 

Adhering to this standard helps us

*   Provide a sense of consistency throughout the library
*   Give users instant access to necessary information at the interpreter prompt (either via the built-in Python function help(object_name) or the Jupyter object_name?)
*   Easily generate a reference manual using sphinx's autodoc and apidoc

It is always useful to build the docs locally before setting up a pull request, and lets you check how your docstrings render in html prior to submitting a pull request.

However once you open a PR a preview of the docs is provided as one of the GitHub Actions.

Further questions
-----------------

We encourage you to reach out to the `QuantEcon team <https://quantecon.org/team>`_ on the 
`Discourse forum <http://discourse.quantecon.org/>`_ if you have any further questions.
