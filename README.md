# QuantEcon.py

A high performance, open source Python code library for economics

```python
  from quantecon.markov import DiscreteDP
  aiyagari_ddp = DiscreteDP(R, Q, beta)
  results = aiyagari_ddp.solve(method='policy_iteration')
```

[![Build Status](https://github.com/QuantEcon/QuantEcon.py/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/QuantEcon.py/actions?query=workflow%3Abuild)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.svg)](https://coveralls.io/r/QuantEcon/QuantEcon.py)
[![Documentation Status](https://readthedocs.org/projects/quanteconpy/badge/?version=latest)](https://quanteconpy.readthedocs.io/en/latest/?badge=latest)

## Installation

Before installing `quantecon` we recommend you install the [Anaconda](https://www.anaconda.com/download/) Python distribution, which includes a full suite of scientific python tools. **Note:** `quantecon` is now only supporting Python version 3.5+. This is mainly to allow code to be written taking full advantage of new features such as using the `@` symbol for matrix multiplication. Therefore please install the latest Python 3 Anaconda distribution.

Next you can install quantecon by opening a terminal prompt and typing

    pip install quantecon

## Usage

Once `quantecon` has been installed you should be able to import it as follows:

```python
import quantecon as qe
```

You can check the version by running

```python
print(qe.__version__)
```

If your version is below what’s available on [PyPI](https://pypi.python.org/pypi/quantecon/) then it is time to upgrade. This can be done by running

    pip install --upgrade quantecon

## Examples and Sample Code

Many examples of QuantEcon.py in action can be found at [Quantitative Economics](https://lectures.quantecon.org/). See also the

*   [Documentation](https://quanteconpy.readthedocs.org/en/latest/)
*   [Notebook gallery](https://notes.quantecon.org)

QuantEcon.py is supported financially by the [Alfred P. Sloan Foundation](http://www.sloan.org/) and is part of the [QuantEcon organization](https://quantecon.org).

## Downloading the `quantecon` Repository

An alternative is to download the sourcecode of the `quantecon` package and install it manually from [the github repository](https://github.com/QuantEcon/QuantEcon.py/). For example, if you have git installed type

    git clone https://github.com/QuantEcon/QuantEcon.py

Once you have downloaded the source files then the package can be installed by running

    pip install flit
    flit install

(To learn the basics about setting up Git see [this link](https://help.github.com/articles/set-up-git/).)
