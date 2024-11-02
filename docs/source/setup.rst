Installation
============

Before installing `quantecon` we recommend you install the [Anaconda](https://www.anaconda.com/download/) Python distribution, which includes a full suite of scientific python tools.

Next you can install quantecon by opening a terminal prompt and typing

.. code:: 
    pip install quantecon


Usage
-----

Once `quantecon` has been installed you should be able to import it as follows:

    import quantecon as qe

You can check the version by running

    print(qe.__version__)

If your version is below what’s available on [PyPI](https://pypi.python.org/pypi/quantecon/) then it is time to upgrade. This can be done by running

    pip install --upgrade quantecon

Downloading the `quantecon` Repository
--------------------------------------

An alternative is to download the sourcecode of the `quantecon` package and install it manually from [the github repository](https://github.com/QuantEcon/QuantEcon.py/). For example, if you have git installed type

    git clone https://github.com/QuantEcon/QuantEcon.py

Once you have downloaded the source files then the package can be installed by running

	cd QuantEcon.py
    pip install ./

(To learn the basics about setting up Git see [this link](https://help.github.com/articles/set-up-git/).)

Examples and Sample Code
------------------------

Many examples of QuantEcon.py in action can be found at [Quantitative Economics](https://lectures.quantecon.org/). 

QuantEcon.py is supported financially by the [Alfred P. Sloan Foundation](http://www.sloan.org/) and is part of the [QuantEcon organization (A NumFOCUS fiscally sponsored project)](/).