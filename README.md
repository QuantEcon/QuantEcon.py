# QuantEcon.py

A high performance, open source Python code library for economics

```python
  from quantecon.markov import DiscreteDP
  aiyagari_ddp = DiscreteDP(R, Q, beta)
  results = aiyagari_ddp.solve(method='policy_iteration')
```

[![Build Status](https://github.com/QuantEcon/QuantEcon.py/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/QuantEcon.py/actions?query=workflow%3Abuild)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.svg)](https://coveralls.io/r/QuantEcon/QuantEcon.py)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://quanteconpy.readthedocs.io/en/stable/)
[![Documentation (latest)](https://img.shields.io/badge/docs-latest-blue.svg)](https://quanteconpy.readthedocs.io/en/latest/)
[![PyPI Version](https://img.shields.io/pypi/v/quantecon.svg)](https://pypi.org/project/quantecon/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/quantecon.svg)](https://pypi.org/project/quantecon/)

## Installation

Before installing `quantecon` we recommend you install the [Anaconda](https://www.anaconda.com/download/) Python distribution, which includes a full suite of scientific python tools.

Next you can install quantecon by opening a terminal prompt and typing

    pip install quantecon

or using conda-forge by typing

    conda install -c conda-forge quantecon

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
*   [Notebook gallery](https://github.com/QuantEcon/notebook-gallery)

QuantEcon.py is supported financially by the [Alfred P. Sloan Foundation](http://www.sloan.org/) and is part of the [QuantEcon organization](https://quantecon.org).

## Downloading the `quantecon` Repository

An alternative is to download the sourcecode of the `quantecon` package and install it manually from [the github repository](https://github.com/QuantEcon/QuantEcon.py/). For example, if you have git installed type

    git clone https://github.com/QuantEcon/QuantEcon.py

Once you have downloaded the source files then the package can be installed by running

    pip install flit
    flit install

(To learn the basics about setting up Git see [this link](https://help.github.com/articles/set-up-git/).)

## Citation

QuantEcon.py is MIT licensed, so you are free to use it without any charge and restriction. If it is convenient for you, please cite QuantEcon.py when using it in your work and also consider contributing all your changes back, so that we can incorporate it.

A BibTeX entry for LaTeX users is

```bibtex
@article{10.21105/joss.05585,
author = {Batista, Quentin and Coleman, Chase and Furusawa, Yuya and Hu, Shu and Lunagariya, Smit and Lyon, Spencer and McKay, Matthew and Oyama, Daisuke and Sargent, Thomas J. and Shi, Zejin and Stachurski, John and Winant, Pablo and Watkins, Natasha and Yang, Ziyue and Zhang, Hengcheng},
doi = {10.5281/zenodo.10345102},
title = {QuantEcon.py: A community based Python library for quantitative economics},
year = {2024},
journal = {Journal of Open Source Software},
volume = {9},
number = {93},
pages = {5585}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quantecon/quantecon.py&type=Date)](https://star-history.com/#quantecon/quantecon.py&Date)
