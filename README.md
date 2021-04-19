
## Quantitative Economics (Python)

A code library for quantitative economic modeling in Python

Library Website: [https://quantecon.org/quantecon-py/](https://quantecon.org/quantecon-py/)

### Installation

See the [library website](https://quantecon.org/quantecon-py/) for instructions

#### Build and Coverage Status:

[![Build Status](https://github.com/QuantEcon/QuantEcon.py/workflows/build/badge.svg)](https://github.com/QuantEcon/QuantEcon.py/actions?query=workflow%3Abuild)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.svg)](https://coveralls.io/r/QuantEcon/QuantEcon.py)
[![Code Quality: Python](https://img.shields.io/lgtm/grade/python/g/QuantEcon/QuantEcon.py.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/QuantEcon/QuantEcon.py/context:python)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/QuantEcon/QuantEcon.py.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/QuantEcon/QuantEcon.py/alerts)

#### ReadTheDocs Status:

[![Documentation Status](https://readthedocs.org/projects/quanteconpy/badge/?version=latest)](https://quanteconpy.readthedocs.io/en/latest/?badge=latest)

#### Gitter

[![Join the chat at https://gitter.im/QuantEcon/QuantEcon.py](https://badges.gitter.im/QuantEcon/QuantEcon.py.svg)](https://gitter.im/QuantEcon/QuantEcon.py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


## Additional Links

1. [Project Coordinators](http://quantecon.org/team)
2. [Lead Developers](http://quantecon.org/team)
3. [QuantEcon Lecture Website](https://lectures.quantecon.org)

### License

Copyright © 2013-2017 Thomas J. Sargent and John Stachurski: BSD-3
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

## Major Changes

For a complete list of changes please refer to the [CHANGELOG.md](CHANGELOG.md)

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


### Ver 0.4.2 (26-November-2018)

1. FEAT: Add AS algorithm. [\#433](https://github.com/QuantEcon/QuantEcon.py/pull/433) ([shizejin](https://github.com/shizejin))
1. FEAT: Add method option in robustlq.py [\#437](https://github.com/QuantEcon/QuantEcon.py/pull/437) ([hinayuki64](https://github.com/hinayuki64))
1. FEAT: Add Player.delete\_action, NormalFormGame.delete\_action [\#444](https://github.com/QuantEcon/QuantEcon.py/pull/444) ([oyamad](https://github.com/oyamad))
1. FEAT: Add the Nelder-Mead algorithm [\#441](https://github.com/QuantEcon/QuantEcon.py/pull/441) ([QBatista](https://github.com/QBatista))
1. FEAT: Added basic inequality mesasures: lorenz curve and gini [\#414](https://github.com/QuantEcon/QuantEcon.py/pull/414) ([cdagnino](https://github.com/cdagnino))
1. MAINT: Remove `from future import ...` [\#436](https://github.com/QuantEcon/QuantEcon.py/pull/436) ([hinayuki64](https://github.com/hinayuki64))
1. FIX: Force tuple elements to have the same dtype [\#435](https://github.com/QuantEcon/QuantEcon.py/pull/435) ([oyamad](https://github.com/oyamad))
1. DOC: fix brent\_max docstring [\#440](https://github.com/QuantEcon/QuantEcon.py/pull/440) ([natashawatkins](https://github.com/natashawatkins))
1. FIX: Disallow Player with 0 actions [\#443](https://github.com/QuantEcon/QuantEcon.py/pull/443) ([oyamad](https://github.com/oyamad))

### Ver 0.4.1 (17-September-2018)

1. FEAT: add solver for dynamic linear economies as LQ problem [\#426](https://github.com/QuantEcon/QuantEcon.py/pull/426) ([mmcky](https://github.com/mmcky))
1. DOC: Fix the doc of `root\_finding.py` to display nicely [\#431](https://github.com/QuantEcon/QuantEcon.py/pull/431) ([QBatista](https://github.com/QBatista))

### Ver 0.4.0 (20-August-2018)

1. FEAT: Add bisection and brent's method for root finding. See PR [\#424](https://github.com/QuantEcon/QuantEcon.py/pull/424) ([spvdchachan](https://github.com/spvdchachan))
1. FEAT: Add `qhull\_options` to `game\_theory.vertex\_enumeration`. See PR [\#421](https://github.com/QuantEcon/QuantEcon.py/pull/421) ([oyamad](https://github.com/oyamad))
1. FEAT: Root finding. See PR [\#417](https://github.com/QuantEcon/QuantEcon.py/pull/417) ([chrishyland](https://github.com/chrishyland))
1. FEAT: Add `'interior-point'` option to `is\_dominated`; add `dominated\_actions`. See PR [\#415](https://github.com/QuantEcon/QuantEcon.py/pull/415) ([oyamad](https://github.com/oyamad))
1. FEAT: Add hamilton filter. See PR [\#405](https://github.com/QuantEcon/QuantEcon.py/pull/405) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
1. FEAT: Add sample game generators from bimatrix-generators. See PR [\#392](https://github.com/QuantEcon/QuantEcon.py/pull/392) ([oyamad](https://github.com/oyamad))
1. MAINT: update to new rtd requirements spec. See PR [\#427](https://github.com/QuantEcon/QuantEcon.py/pull/427) ([mmcky](https://github.com/mmcky))
1. MAINT: Add `requests` to setup.py. See PR [\#420](https://github.com/QuantEcon/QuantEcon.py/pull/420) ([oyamad](https://github.com/oyamad))
1. MAINT: Add `mock` to the dependencies list. See PR [\#418](https://github.com/QuantEcon/QuantEcon.py/pull/418) ([oyamad](https://github.com/oyamad))
1. TEST: Fix test\_discrete\_rv. See PR [\#412](https://github.com/QuantEcon/QuantEcon.py/pull/412) ([oyamad](https://github.com/oyamad))
1. MAINT: add minimum version number for numba support. See PR [\#409](https://github.com/QuantEcon/QuantEcon.py/pull/409) ([mmcky](https://github.com/mmcky))
1. MAINT: Setup an auto-generate changelog for releases. See PR [\#403](https://github.com/QuantEcon/QuantEcon.py/pull/403) ([mmcky](https://github.com/mmcky))

### Ver 0.3.8 (14-March-2018)
1. FEAT: Add random.draw. See [PR #397](https://github.com/QuantEcon/QuantEcon.py/pull/397)
1. FEAT: Add Numba jit version of scipy.special.comb. See [PR #377](https://github.com/QuantEcon/QuantEcon.py/pull/377)
1. FEAT: Add random_tournament_graph for game theory module. See [PR #378](https://github.com/QuantEcon/QuantEcon.py/pull/378)
2. MAINT: Implement Sigma_infinity and K_infinity as properties. See [PR #396](https://github.com/QuantEcon/QuantEcon.py/pull/396)
2. MAINT: Use `np.ix_` to extract submatrix. See [PR #389](https://github.com/QuantEcon/QuantEcon.py/pull/389)
2. MAINT: support_enumeration: Refactoring. See [PR #384](https://github.com/QuantEcon/QuantEcon.py/pull/384)
2. MAINT: pure_nash_brute: Add tol option. See [PR #385](https://github.com/QuantEcon/QuantEcon.py/pull/385)
2. MAINT: NormalFormGame: Add `payoff_arrays` attribute. See [PR #382](https://github.com/QuantEcon/QuantEcon.py/pull/382)
2. MAINT: Re-implement `next_k_array`; add `k_array_rank`. See [PR #379](https://github.com/QuantEcon/QuantEcon.py/pull/379)
3. FIX: Fix tac, toc, loop_timer to return float. See [PR #387](https://github.com/QuantEcon/QuantEcon.py/pull/387)
3. FIX: Update to ``scipy.special.com``. See [PR #375](https://github.com/QuantEcon/QuantEcon.py/pull/375)
4. DEPRECATE: remove models subpackage. See [PR #383](https://github.com/QuantEcon/QuantEcon.py/pull/383)
5. DOCS: Improvements to documentation. See [PR #388](https://github.com/QuantEcon/QuantEcon.py/pull/388)

Contributors: [oyamad](https://github.com/oyamad), [QBatista](https://github.com/QBatista), [mcsalgado](https://github.com/mcsalgado), and [okuchap](https://github.com/okuchap)

### Ver 0.3.7 (01-November-2017)
1. FEAT: Add random_state option to arma.py with tests. See [PR #329](https://github.com/QuantEcon/QuantEcon.py/pull/329)
2. FEAT: New features for timing functions. See [PR #340](https://github.com/QuantEcon/QuantEcon.py/pull/340)
3. Improved test coverage ([PR #343](https://github.com/QuantEcon/QuantEcon.py/pull/343))
4. FEAT: Add option to supply a random seed for discrete_rv, lqcontrol, lqnash, lss, and quad ([PR #346](https://github.com/QuantEcon/QuantEcon.py/pull/346))
5. FIX: RBLQ: add pure forecasting case ([PR #355](https://github.com/QuantEcon/QuantEcon.py/pull/355))
6. FEAT: jit the 1d quadrature routines ([PR #352](https://github.com/QuantEcon/QuantEcon.py/pull/352))
7. FIX: Replace `np.isfinite(cn)` with `cn * EPS < 1` ([PR #361](https://github.com/QuantEcon/QuantEcon.py/pull/361))
8. FEAT: Add option to `solve_discrete_riccati` to use `scipy.linalg.solve_discrete_are` ([PR #362](https://github.com/QuantEcon/QuantEcon.py/pull/362))
9. FIX: Bugfix to `solve_discrete_riccati` ([PR #364](https://github.com/QuantEcon/QuantEcon.py/pull/364))
10. Minor Fixes ([PR #342](https://github.com/QuantEcon/QuantEcon.py/pull/342))

### Ver 0.3.6.2 (27-August-2017)
1. FIX: support_enumeration: Use ``_numba_linalg_solve``. See [PR #311](https://github.com/QuantEcon/QuantEcon.py/pull/311)
2. Updated Docstrings for better math rendering. See [PR #315](https://github.com/QuantEcon/QuantEcon.py/pull/315)
3. ENH: added routines to convert ddp between full and SA formulations. See [PR #318](https://github.com/QuantEcon/QuantEcon.py/pull/318)
4. Added tests for Distributions. See [PR #324](https://github.com/QuantEcon/QuantEcon.py/pull/324)
5. Added tests for lemke howson exceptions. See [PR #323](https://github.com/QuantEcon/QuantEcon.py/pull/323)
6. Added vertex_enumeration to game theory module. See [PR #326](https://github.com/QuantEcon/QuantEcon.py/pull/326)
7. Added ``is_dominated`` method to game_theory.player. See [PR #327](https://github.com/QuantEcon/QuantEcon.py/pull/327)
8. Minor Updates ([PR #320](https://github.com/QuantEcon/QuantEcon.py/pull/320), [PR #321](https://github.com/QuantEcon/QuantEcon.py/pull/321),
[PR #328](https://github.com/QuantEcon/QuantEcon.py/pull/328))

### Ver 0.3.5.1 (17-May-2017)
1. Add rouwenhorst method for approx AR(1) with MC. See [PR #282](https://github.com/QuantEcon/QuantEcon.py/pull/282)
2. Added tests to improve coverage ([PR #282](https://github.com/QuantEcon/QuantEcon.py/pull/282),
[PR #303](https://github.com/QuantEcon/QuantEcon.py/pull/303), [PR #309](https://github.com/QuantEcon/QuantEcon.py/pull/309))
3. Minor Fixes ([PR #296](https://github.com/QuantEcon/QuantEcon.py/pull/296), [PR #297](https://github.com/QuantEcon/QuantEcon.py/pull/297))

### Ver. 0.3.4 (23-February-2017)
1. Add support_enumeration, a simple algorithm that computes all mixed-action Nash equilibria of a non-degenerate 2-player game. See [PR #263](https://github.com/QuantEcon/QuantEcon.py/pull/263)
2. Various fixes for issues with numba. See [PR #265](https://github.com/QuantEcon/QuantEcon.py/pull/265), [PR #283](https://github.com/QuantEcon/QuantEcon.py/pull/283)
3. Add lemke_howson algorithm to game_theory module. See [PR #268](https://github.com/QuantEcon/QuantEcon.py/pull/268)
4. Add random game generators to game_theory module. See [PR #270](https://github.com/QuantEcon/QuantEcon.py/pull/270)
5. Implement the imitation game algorithm by McLennan and Tourky. See [PR #273](https://github.com/QuantEcon/QuantEcon.py/pull/273)
6. Add brute force for finding pure nash equilibria. See [PR #276](https://github.com/QuantEcon/QuantEcon.py/pull/276)
7. Improve parameter names to QuantEcon.notebooks dependency fetcher. See [PR #279](https://github.com/QuantEcon/QuantEcon.py/pull/279)
8. Utilities ``tic``, ``tac`` and ``toc`` moved to top level namespace of package. See [PR #280](https://github.com/QuantEcon/QuantEcon.py/pull/280)

### Ver. 0.3.3 (21-July-2016)
1. Remove ``python2.7`` classifiers project only supports ``python3.5+``
2. Migrate ``sa_indices`` to be a utility function for the markov submodule
3. Updates ``probvec`` to include a multi-core parallel option using numba infrastructure in ``quantecon/random/utilities.py``

### Ver. 0.3.2 (25-April-2016)

1. Minor changes to ``NormalFormGame``. See [PR #226](https://github.com/QuantEcon/QuantEcon.py/pull/226)
2. Update ``tauchen`` code to make use of Numba. See [PR #227](https://github.com/QuantEcon/QuantEcon.py/pull/227)
3. Remove ``Python 2.7`` from test environment. Will support Python 3.5+
4. Updated ``qe.util.nb_fetch`` to not overwrite files by default
6. Remove ``num_actions`` from DiscreteDP. See [PR #236](https://github.com/QuantEcon/QuantEcon.py/pull/236)
7. Add states/nodes to ``MarkovChain``/``DiGraph``. See [PR #237](https://github.com/QuantEcon/QuantEcon.py/pull/237)
8. Updated ``DiscreteDP`` to include ``backward_induction`` (DiscreteDP now accepts beta=1). See [PR #244](https://github.com/QuantEcon/QuantEcon.py/pull/244)
9. ``Numba`` is now a formal dependency.
10. Modified ``tauchen`` to return a ``MarkovChain`` instance. See [PR #250](https://github.com/QuantEcon/QuantEcon.py/pull/250)

### Ver. 0.3.1 (22-January-2016)

1. Adds the ``quantecon/game_theory/`` sub package
2. Updates api for using ``distributions`` as a module ``qe.distributions``

### Ver. 0.3

1. Removes ``quantecon/models`` subpackage and the collection of code examples. Code has been migrated to the [QuantEcon.applications](https://github.com/QuantEcon/QuantEcon.applications) repository.
2. Adds a utility for fetching notebook dependencies from [QuantEcon.applications](https://github.com/QuantEcon/QuantEcon.applications) to support community contributed notebooks.
