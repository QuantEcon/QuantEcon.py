# QuantEcon.py

A high performance, open source Python code library for economics

  from quantecon.markov import DiscreteDP
  aiyagari_ddp = DiscreteDP(R, Q, beta)
  results = aiyagari_ddp.solve(method='policy_iteration')

[![Build Status](https://github.com/QuantEcon/QuantEcon.py/workflows/build/badge.svg)](https://github.com/QuantEcon/QuantEcon.py/actions?query=workflow%3Abuild)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.svg)](https://coveralls.io/r/QuantEcon/QuantEcon.py)
[![Code Quality: Python](https://img.shields.io/lgtm/grade/python/g/QuantEcon/QuantEcon.py.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/QuantEcon/QuantEcon.py/context:python)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/QuantEcon/QuantEcon.py.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/QuantEcon/QuantEcon.py/alerts)
[![Documentation Status](https://readthedocs.org/projects/quanteconpy/badge/?version=latest)](https://quanteconpy.readthedocs.io/en/latest/?badge=latest)

<<<<<<< HEAD
#### Gitter

[![Join the chat at https://gitter.im/QuantEcon/QuantEcon.py](https://badges.gitter.im/QuantEcon/QuantEcon.py.svg)](https://gitter.im/QuantEcon/QuantEcon.py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


## Additional Links

1. [Project Coordinators](http://quantecon.org/team)
2. [Lead Developers](http://quantecon.org/team)
3. [QuantEcon Lecture Website](https://lectures.quantecon.org)

## Major Changes

For a complete list of changes please refer to the [CHANGELOG.md](CHANGELOG.md)

## Ver 0.5.2 (16-November-2021)

This is a bug fix release

**Maintain:**

1. FIX: [markov: Respect dtype of P in cdfs](https://github.com/QuantEcon/QuantEcon.py/pull/592) ([[oyamad](https://github.com/oyamad)], thanks [@btanner](https://github.com/btanner) for reporting issue)
2. [LGTM code quality suggestions](https://github.com/QuantEcon/QuantEcon.py/pull/588) ([nshea3](https://github.com/QuantEcon/QuantEcon.py/pull/588))

## Ver 0.5.1 (27-June-2021)

**New:**

1. ENH: [Add Numba-jitted linprog solver](https://github.com/QuantEcon/QuantEcon.py/pull/532) ([[oyamad](https://github.com/oyamad)])
2. EHN: [Add minmax solver](https://github.com/QuantEcon/QuantEcon.py/pull/579) ([[oyamad](https://github.com/oyamad)])
3. ENH: [Add LP solution method to DiscreteDP](https://github.com/QuantEcon/QuantEcon.py/pull/585) ([[oyamad](https://github.com/oyamad)])

**Maintain:**

1. MAINT: [Use multivariate_normal via random_state](https://github.com/QuantEcon/QuantEcon.py/pull/581) ([[oyamad](https://github.com/oyamad)])
2. FIX: [minmax: Fix redundancy](https://github.com/QuantEcon/QuantEcon.py/pull/582) ([[oyamad](https://github.com/oyamad)])
3. DOCS: [Fix typos in Docs](https://github.com/QuantEcon/QuantEcon.py/pull/584) ([[timgates42](https://github.com/timgates42)])

### Ver 0.5.0 (19-April-2021)

**Breaking Changes:**

1. ENH: Extend `LinearStateSpace` class [\#569](https://github.com/QuantEcon/QuantEcon.py/pull/569) ([shizejin](https://github.com/shizejin))

**Other Changes:**

- FIX: [kalman] Always initialize self.Sigma and self.x\_hat [\#562](https://github.com/QuantEcon/QuantEcon.py/pull/562) ([rht](https://github.com/rht))
- TST: Setup Tests via Github Actions [\#561](https://github.com/QuantEcon/QuantEcon.py/pull/561) ([rht](https://github.com/rht))
- ENH: Update root\_finding.py [\#560](https://github.com/QuantEcon/QuantEcon.py/pull/560) ([alanlujan91](https://github.com/alanlujan91))

Special thanks for contributions by [rht](https://github.com/rht), [shizejin](https://github.com/shizejin), [alanlujan91](https://github.com/alanlujan91), and [oyamad](https://github.com/oyamad)


### Ver 0.4.8 (02-July-2020)

- FIX: rank-size test by inc. sample size [\#556](https://github.com/QuantEcon/QuantEcon.py/pull/556) ([bktaha](https://github.com/bktaha))
- REF and TEST: rank\_size in inequality.py [\#551](https://github.com/QuantEcon/QuantEcon.py/pull/551) ([bktaha](https://github.com/bktaha))
- FIX: ValueError `LQMarkov` convergence failed, Closes \#508 [\#550](https://github.com/QuantEcon/QuantEcon.py/pull/550) ([bktaha](https://github.com/bktaha))
- rank\_size\_plot\_typo [\#545](https://github.com/QuantEcon/QuantEcon.py/pull/545) ([shlff](https://github.com/shlff))
- Fix variables never used lgtm warnings in dle.py. [\#542](https://github.com/QuantEcon/QuantEcon.py/pull/542) ([duncanhobbs](https://github.com/duncanhobbs))
- Fix lgtm warnings in quadsums.py. [\#541](https://github.com/QuantEcon/QuantEcon.py/pull/541) ([duncanhobbs](https://github.com/duncanhobbs))
- Fix lgtm warning for arma.py. [\#540](https://github.com/QuantEcon/QuantEcon.py/pull/540) ([duncanhobbs](https://github.com/duncanhobbs))

Special thanks for contributions by [bktaha](https://github.com/bktaha), [duncanhobbs](https://github.com/duncanhobbs), and [shlff](https://github.com/shlff).

### Ver 0.4.7 (24-Apr-2020)

1. FIX: Updates for Numba 0.49.0 [\#531](https://github.com/QuantEcon/QuantEcon.py/pull/531) ([oyamad](https://github.com/oyamad))
1. FIX: a link on README [\#529](https://github.com/QuantEcon/QuantEcon.py/pull/529) ([oyamad](https://github.com/oyamad))
1. UPD: Remove unused variable [\#526](https://github.com/QuantEcon/QuantEcon.py/pull/526) ([MKobayashi23m](https://github.com/MKobayashi23m))
1. UPD: bimatrix\_generators: Define `\_\_all\_\_` [\#525](https://github.com/QuantEcon/QuantEcon.py/pull/525) ([oyamad](https://github.com/oyamad))
1. UPD: remove old test commands from Makefile [\#524](https://github.com/QuantEcon/QuantEcon.py/pull/524) ([mmcky](https://github.com/mmcky))

### Ver 0.4.6 (09-December-2019)

1. FEAT: Adds a rank size plot to inequality [\#518](https://github.com/QuantEcon/QuantEcon.py/pull/518) ([jstac](https://github.com/jstac))
1. UPD: General cleanup of Package [\#515](https://github.com/QuantEcon/QuantEcon.py/pull/515) ([mmcky](https://github.com/mmcky))
1. \[FIX\] Fix Future Warnings in ivp.py and test\_quad.py and RuntimeError in lq\_control.py. [\#509](https://github.com/QuantEcon/QuantEcon.py/pull/509) ([duncanhobbs](https://github.com/duncanhobbs))
1. FIX: Player.is\_dominated: Fix warnings [\#504](https://github.com/QuantEcon/QuantEcon.py/pull/504) ([oyamad](https://github.com/oyamad))
1. FIX: random.draw: Replace `random\_sample` with `random` [\#503](https://github.com/QuantEcon/QuantEcon.py/pull/503) ([oyamad](https://github.com/oyamad))
1. FIX: two minor modifications in `lqcontrol` [\#498](https://github.com/QuantEcon/QuantEcon.py/pull/498) ([shizejin](https://github.com/shizejin))
1. UPD: Update travis to use python=3.7 [\#494](https://github.com/QuantEcon/QuantEcon.py/pull/494) ([mmcky](https://github.com/mmcky))

### Ver 0.4.5 (08-July-2019)

1. ENH: Add `LQMarkov`. [\#489](https://github.com/QuantEcon/QuantEcon.py/pull/489) ([shizejin](https://github.com/shizejin))
1. FIX: Increase `tol` in `rouwenhorst` test. [\#492](https://github.com/QuantEcon/QuantEcon.py/pull/492) ([shizejin](https://github.com/shizejin)) to fix [\#491](https://github.com/QuantEcon/QuantEcon.py/issues/491)
1. TRAVIS: Set coverage branch as `linux`. [\#490](https://github.com/QuantEcon/QuantEcon.py/pull/490) ([shizejin](https://github.com/shizejin))
1. FIX: DOC: Remove `matplotlib.sphinxext.only\_directives` [\#488](https://github.com/QuantEcon/QuantEcon.py/pull/488) ([oyamad](https://github.com/oyamad))

### Ver 0.4.4 (24-May-2019)

1. FEAT: Add drift term keyword to `markov.tauchen`. [\#484](https://github.com/QuantEcon/QuantEcon.py/pull/484) ([shizejin](https://github.com/shizejin))
1. FIX: Import scipy.sparse.linalg [\#482](https://github.com/QuantEcon/QuantEcon.py/pull/482) ([oyamad](https://github.com/oyamad))
1. FIX: `sample\_without\_replacement` using guvectorize [\#479](https://github.com/QuantEcon/QuantEcon.py/pull/479) ([oyamad](https://github.com/oyamad))
1. FEAT: Add `random\_pure\_actions` and `random\_mixed\_actions` [\#477](https://github.com/QuantEcon/QuantEcon.py/pull/477) ([okuchap](https://github.com/okuchap))
1. FIX: Raise correct error when `A` is not square in `LinearStateSpace` [\#475](https://github.com/QuantEcon/QuantEcon.py/pull/475) ([QBatista](https://github.com/QBatista))
1. FIX: alerts by lgtm [\#474](https://github.com/QuantEcon/QuantEcon.py/pull/474) ([okuchap](https://github.com/okuchap))
1. Fix flake8 errors [\#470](https://github.com/QuantEcon/QuantEcon.py/pull/470) ([rht](https://github.com/rht))
1. TEST: Fix the names of tests for `brent\_max` [\#469](https://github.com/QuantEcon/QuantEcon.py/pull/469) ([QBatista](https://github.com/QBatista))
1. DOC: Update example for `nelder\_mead` [\#468](https://github.com/QuantEcon/QuantEcon.py/pull/468) ([QBatista](https://github.com/QBatista))
1. FIX: all F401 unused imports [\#467](https://github.com/QuantEcon/QuantEcon.py/pull/467) ([rht](https://github.com/rht))

### Ver 0.4.3 (17-December-2018)

1.  INFRA: Isolate rtd-specific requirements to doc-requirements.txt [\#464](https://github.com/QuantEcon/QuantEcon.py/pull/464) ([rht](https://github.com/rht))
1. DOCS: fix for lorenz documentation [\#462](https://github.com/QuantEcon/QuantEcon.py/pull/462) ([natashawatkins](https://github.com/natashawatkins))
1. INFRA: Disable performance tests [\#461](https://github.com/QuantEcon/QuantEcon.py/pull/461) ([rht](https://github.com/rht))
1. ENH: quad: Import sympy only when necessary [\#459](https://github.com/QuantEcon/QuantEcon.py/pull/459) ([rht](https://github.com/rht))
1. INFRA: Travis: Move dependency installs with wheels available to pip [\#458](https://github.com/QuantEcon/QuantEcon.py/pull/458) ([rht](https://github.com/rht))
1. DOCS: Update Documentation [\#454](https://github.com/QuantEcon/QuantEcon.py/pull/454) ([mmcky](https://github.com/mmcky))
1. README: Update coveralls badge to use svg [\#453](https://github.com/QuantEcon/QuantEcon.py/pull/453) ([rht](https://github.com/rht))
1. FIX: Fix warning in test\_pure\_nash [\#451](https://github.com/QuantEcon/QuantEcon.py/pull/451) ([oyamad](https://github.com/oyamad))
1. ENH: Add errors for invalid inputs for `brent\_max` [\#450](https://github.com/QuantEcon/QuantEcon.py/pull/450) ([QBatista](https://github.com/QBatista))
1. INFRA: Travis: Add macOS to the build matrix [\#448](https://github.com/QuantEcon/QuantEcon.py/pull/448) ([rht](https://github.com/rht))
1. FEAT: Add Shorrocks mobility index [\#447](https://github.com/QuantEcon/QuantEcon.py/pull/447) ([natashawatkins](https://github.com/natashawatkins))
1. FIX: test `method` keyword of `RepeatedGame.equilibrium\_payoffs\(\)`. [\#446](https://github.com/QuantEcon/QuantEcon.py/pull/446) ([shizejin](https://github.com/shizejin))
=======
## Installation
>>>>>>> master

Before installing `quantecon` we recommend you install the [Anaconda](https://www.anaconda.com/download/) Python distribution, which includes a full suite of scientific python tools. **Note:** quantecon is now only supporting Python version 3.5+. This is mainly to allow code to be written taking full advantage of new features such as using the @ symbol for matrix multiplication. Therefore please install the latest Python 3 Anaconda distribution.

Next you can install quantecon by opening a terminal prompt and typing

    pip install quantecon

## Usage

Once `quantecon` has been installed you should be able to import it as follows:

    import quantecon as qe

You can check the version by running

    print(qe.__version__)

If your version is below what’s available on [PyPI](https://pypi.python.org/pypi/quantecon/) then it is time to upgrade. This can be done by running

    pip install --upgrade quantecon

## Examples and Sample Code

Many examples of QuantEcon.py in action can be found at [Quantitative Economics](https://lectures.quantecon.org/). See also the

*   [Documentation](http://quanteconpy.readthedocs.org/en/latest/)
*   [Notebook gallery](/notebooks)
*   [Additional Examples](/python-examples)

QuantEcon.py is supported financially by the [Alfred P. Sloan Foundation](http://www.sloan.org/) and is part of the [QuantEcon organization](/).

## Downloading the `quantecon` Repository

An alternative is to download the sourcecode of the `quantecon` package and install it manually from [the github repository](https://github.com/QuantEcon/QuantEcon.py/). For example, if you have git installed type

    git clone https://github.com/QuantEcon/QuantEcon.py

Once you have downloaded the source files then the package can be installed by running

    python setup.py install

(To learn the basics about setting up Git see [this link](https://help.github.com/articles/set-up-git/).)