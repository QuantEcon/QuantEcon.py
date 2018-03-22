# Change Log

## [Unreleased](https://github.com/QuantEcon/QuantEcon.py/tree/HEAD)

[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.8.1...HEAD)

**Merged pull requests:**

- README: Add a contributor [\#401](https://github.com/QuantEcon/QuantEcon.py/pull/401) ([oyamad](https://github.com/oyamad))
- Add sample game generators from bimatrix-generators [\#392](https://github.com/QuantEcon/QuantEcon.py/pull/392) ([oyamad](https://github.com/oyamad))

## [0.3.8.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.8.1) (2018-03-13)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.8...0.3.8.1)

## [0.3.8](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.8) (2018-03-12)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.7...0.3.8)

**Fixed bugs:**

- interp in NumPy/SciPy [\#189](https://github.com/QuantEcon/QuantEcon.py/issues/189)

**Closed issues:**

- BUGs in random\_choice [\#393](https://github.com/QuantEcon/QuantEcon.py/issues/393)
- Replace DiscreteRV with jitted function [\#390](https://github.com/QuantEcon/QuantEcon.py/issues/390)
- Return value of timing functions [\#386](https://github.com/QuantEcon/QuantEcon.py/issues/386)
- pure\_nash\_brute: Add `tol` option [\#381](https://github.com/QuantEcon/QuantEcon.py/issues/381)
- Remove reference to `models`? [\#380](https://github.com/QuantEcon/QuantEcon.py/issues/380)
- \[Style\] Use of unicode as arguments and keyword arguments? [\#373](https://github.com/QuantEcon/QuantEcon.py/issues/373)
- Setup new release of QuantEcon.py [\#365](https://github.com/QuantEcon/QuantEcon.py/issues/365)
- Migrate to using setuptools rather than distutils for package setup [\#304](https://github.com/QuantEcon/QuantEcon.py/issues/304)
- Remove dependence on matplotlib [\#262](https://github.com/QuantEcon/QuantEcon.py/issues/262)

**Merged pull requests:**

- FEAT: Add random.draw [\#397](https://github.com/QuantEcon/QuantEcon.py/pull/397) ([oyamad](https://github.com/oyamad))
- Implement Sigma\_infinity and K\_infinity as properties [\#396](https://github.com/QuantEcon/QuantEcon.py/pull/396) ([mcsalgado](https://github.com/mcsalgado))
- FEAT: Add jitted function for drawing [\#391](https://github.com/QuantEcon/QuantEcon.py/pull/391) ([QBatista](https://github.com/QBatista))
- MAINT: Use `np.ix\_` to extract submatrix [\#389](https://github.com/QuantEcon/QuantEcon.py/pull/389) ([oyamad](https://github.com/oyamad))
- DOC: Generate doc file for util/combinatorics [\#388](https://github.com/QuantEcon/QuantEcon.py/pull/388) ([oyamad](https://github.com/oyamad))
-  Fix tac, toc, loop\_timer to return float [\#387](https://github.com/QuantEcon/QuantEcon.py/pull/387) ([oyamad](https://github.com/oyamad))
- pure\_nash\_brute: Add tol option [\#385](https://github.com/QuantEcon/QuantEcon.py/pull/385) ([okuchap](https://github.com/okuchap))
- support\_enumeration: Refactoring [\#384](https://github.com/QuantEcon/QuantEcon.py/pull/384) ([oyamad](https://github.com/oyamad))
- remove models subpackage from QuantEcon.py [\#383](https://github.com/QuantEcon/QuantEcon.py/pull/383) ([mmcky](https://github.com/mmcky))
- NormalFormGame: Add `payoff\_arrays` attribute [\#382](https://github.com/QuantEcon/QuantEcon.py/pull/382) ([oyamad](https://github.com/oyamad))
- Re-implement `next\_k\_array`; add `k\_array\_rank` [\#379](https://github.com/QuantEcon/QuantEcon.py/pull/379) ([oyamad](https://github.com/oyamad))
- Add random\_tournament\_graph [\#378](https://github.com/QuantEcon/QuantEcon.py/pull/378) ([oyamad](https://github.com/oyamad))
- Add Numba jit version of scipy.special.comb [\#377](https://github.com/QuantEcon/QuantEcon.py/pull/377) ([oyamad](https://github.com/oyamad))
- FIX: Update `num\_compositions` [\#375](https://github.com/QuantEcon/QuantEcon.py/pull/375) ([QBatista](https://github.com/QBatista))
- Add bugfix PR 364 to Major Changes for v0.3.7 [\#372](https://github.com/QuantEcon/QuantEcon.py/pull/372) ([oyamad](https://github.com/oyamad))

## [0.3.7](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.7) (2017-11-01)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.6.2...0.3.7)

**Implemented enhancements:**

- Daily Build Server for checking examples/\*.py and solutions/\*.ipynb [\#178](https://github.com/QuantEcon/QuantEcon.py/issues/178)

**Fixed bugs:**

- BUG: solve\_discrete\_riccati returns non-stabilizing solution [\#356](https://github.com/QuantEcon/QuantEcon.py/issues/356)
- Possible bug: dare\_test\_tjm\_2 and dare\_test\_tjm\_3 with accelerate/mkl [\#84](https://github.com/QuantEcon/QuantEcon.py/issues/84)

**Closed issues:**

- qe.solve\_discrete\_riccati versus scipy.linalg.solve\_discrete\_are [\#360](https://github.com/QuantEcon/QuantEcon.py/issues/360)
- Problem with install QuantEcon [\#341](https://github.com/QuantEcon/QuantEcon.py/issues/341)
- Release latest version to PyPI [\#332](https://github.com/QuantEcon/QuantEcon.py/issues/332)
- Remove plotting functionality from ARMA [\#325](https://github.com/QuantEcon/QuantEcon.py/issues/325)
- New features for timing functions [\#322](https://github.com/QuantEcon/QuantEcon.py/issues/322)
- TEST: add tests for cartesian.py [\#299](https://github.com/QuantEcon/QuantEcon.py/issues/299)
- TEST: Add tests for arma.py [\#298](https://github.com/QuantEcon/QuantEcon.py/issues/298)
- Problems with `fetch\_nb\_dependency` [\#278](https://github.com/QuantEcon/QuantEcon.py/issues/278)
- Option to supply a random seed [\#153](https://github.com/QuantEcon/QuantEcon.py/issues/153)

**Merged pull requests:**

- update version numbers for new release 0.3.7 [\#371](https://github.com/QuantEcon/QuantEcon.py/pull/371) ([mmcky](https://github.com/mmcky))
- add ability to use setuptools in preference to distutils.core [\#369](https://github.com/QuantEcon/QuantEcon.py/pull/369) ([mmcky](https://github.com/mmcky))
- FIX: Remove 0.0 from `candidates` in `solve\_discrete\_riccati` [\#364](https://github.com/QuantEcon/QuantEcon.py/pull/364) ([oyamad](https://github.com/oyamad))
- Riccati: Add option to use scipy.linalg.solve\_discrete\_are [\#362](https://github.com/QuantEcon/QuantEcon.py/pull/362) ([oyamad](https://github.com/oyamad))
- FIX: Replace `np.isfinite\(cn\)` with `cn \* EPS \< 1` in solve\_discrete\_riccati [\#361](https://github.com/QuantEcon/QuantEcon.py/pull/361) ([oyamad](https://github.com/oyamad))
- TRAVIS: Update Python version to 3.6 [\#358](https://github.com/QuantEcon/QuantEcon.py/pull/358) ([oyamad](https://github.com/oyamad))
- TEST: Set atol [\#357](https://github.com/QuantEcon/QuantEcon.py/pull/357) ([oyamad](https://github.com/oyamad))
- RBLQ: add pure forecasting case  [\#355](https://github.com/QuantEcon/QuantEcon.py/pull/355) ([szokeb87](https://github.com/szokeb87))
- fix kalman class docstring [\#353](https://github.com/QuantEcon/QuantEcon.py/pull/353) ([natashawatkins](https://github.com/natashawatkins))
- ENH: jit the 1d quadrature routines [\#352](https://github.com/QuantEcon/QuantEcon.py/pull/352) ([sglyon](https://github.com/sglyon))
- remove cartesian as has now been migrated to graph\_tools [\#351](https://github.com/QuantEcon/QuantEcon.py/pull/351) ([mmcky](https://github.com/mmcky))
- DOC: Complete docstring for gridmake and \_gridmake2 [\#348](https://github.com/QuantEcon/QuantEcon.py/pull/348) ([QBatista](https://github.com/QBatista))
- FEAT: Add option to supply a random seed \(issue \#153\) [\#346](https://github.com/QuantEcon/QuantEcon.py/pull/346) ([QBatista](https://github.com/QBatista))
- Add simplex\_grid and simplex\_index [\#344](https://github.com/QuantEcon/QuantEcon.py/pull/344) ([oyamad](https://github.com/oyamad))
- TEST: Increase coverage on game\_theory [\#343](https://github.com/QuantEcon/QuantEcon.py/pull/343) ([QBatista](https://github.com/QBatista))
- DOC: Fix a math definition in docstring [\#342](https://github.com/QuantEcon/QuantEcon.py/pull/342) ([oyamad](https://github.com/oyamad))
- FEAT: New features for timing functions \(Issue \#322\) [\#340](https://github.com/QuantEcon/QuantEcon.py/pull/340) ([QBatista](https://github.com/QBatista))
- Minor fix in README.md [\#339](https://github.com/QuantEcon/QuantEcon.py/pull/339) ([oyamad](https://github.com/oyamad))
- updates to README and version history [\#338](https://github.com/QuantEcon/QuantEcon.py/pull/338) ([mmcky](https://github.com/mmcky))
- TEST: Set 'slow' on slow tests in test\_gridtools.py [\#334](https://github.com/QuantEcon/QuantEcon.py/pull/334) ([oyamad](https://github.com/oyamad))
- coveragerc: Add `@generated\_jit` and `@guvectorize` [\#333](https://github.com/QuantEcon/QuantEcon.py/pull/333) ([oyamad](https://github.com/oyamad))
- FEAT: Add random\_state option \(issue \#153\) [\#329](https://github.com/QuantEcon/QuantEcon.py/pull/329) ([QBatista](https://github.com/QBatista))

## [0.3.6.2](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.6.2) (2017-08-28)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.6.1...0.3.6.2)

**Merged pull requests:**

- issue release of version 0.3.6.2 due to issues with 0.3.6.1 in setupt… [\#337](https://github.com/QuantEcon/QuantEcon.py/pull/337) ([mmcky](https://github.com/mmcky))
- rtd needs matplotlib for some sphinx extensions to work properly [\#336](https://github.com/QuantEcon/QuantEcon.py/pull/336) ([mmcky](https://github.com/mmcky))

## [0.3.6.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.6.1) (2017-08-28)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.6...0.3.6.1)

**Merged pull requests:**

- update to version 0.3.6 for new release to pypi and conda-forge [\#335](https://github.com/QuantEcon/QuantEcon.py/pull/335) ([mmcky](https://github.com/mmcky))

## [0.3.6](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.6) (2017-08-28)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.5.1...0.3.6)

**Implemented enhancements:**

- Improve notebook dependencies fetch utility [\#234](https://github.com/QuantEcon/QuantEcon.py/issues/234)

**Closed issues:**

- Fix: Tic/Tac/Toc Utility [\#314](https://github.com/QuantEcon/QuantEcon.py/issues/314)
- Fix doc warning [\#312](https://github.com/QuantEcon/QuantEcon.py/issues/312)
- Pandas routine 'ols' removed. Move to 'statsmodels' [\#307](https://github.com/QuantEcon/QuantEcon.py/issues/307)
- TEST: Add tests for game\_theory/lemke\_howson.py [\#302](https://github.com/QuantEcon/QuantEcon.py/issues/302)
- TEST: Add tests for distributions.py [\#301](https://github.com/QuantEcon/QuantEcon.py/issues/301)
- Restart ``stable`` in readthedocs when releasing 0.3.5+ [\#290](https://github.com/QuantEcon/QuantEcon.py/issues/290)
- Reintroduce ``cache=True`` for jit compiler in support\_enumeration [\#285](https://github.com/QuantEcon/QuantEcon.py/issues/285)

**Merged pull requests:**

- Remove matplotlib from config files [\#331](https://github.com/QuantEcon/QuantEcon.py/pull/331) ([oyamad](https://github.com/oyamad))
- removed dep on matplotlib [\#330](https://github.com/QuantEcon/QuantEcon.py/pull/330) ([jstac](https://github.com/jstac))
- FIX: Use relative tolerance in test\_tic\_tac\_toc [\#328](https://github.com/QuantEcon/QuantEcon.py/pull/328) ([oyamad](https://github.com/oyamad))
- game\_theory.Player: Add is\_dominated [\#327](https://github.com/QuantEcon/QuantEcon.py/pull/327) ([oyamad](https://github.com/oyamad))
- game\_theory: Add vertex\_enumeration [\#326](https://github.com/QuantEcon/QuantEcon.py/pull/326) ([oyamad](https://github.com/oyamad))
- Tests for distributions.py [\#324](https://github.com/QuantEcon/QuantEcon.py/pull/324) ([QBatista](https://github.com/QBatista))
- Test lemke howson exceptions [\#323](https://github.com/QuantEcon/QuantEcon.py/pull/323) ([QBatista](https://github.com/QBatista))
- Fix class style [\#321](https://github.com/QuantEcon/QuantEcon.py/pull/321) ([lbui01](https://github.com/lbui01))
- add some examples for the documentation on the nb\_fetch utility [\#320](https://github.com/QuantEcon/QuantEcon.py/pull/320) ([mmcky](https://github.com/mmcky))
- fix timing utility docstrings for tic toc tac [\#319](https://github.com/QuantEcon/QuantEcon.py/pull/319) ([mmcky](https://github.com/mmcky))
- ENH: added routines to convert ddp between full and SA formulations [\#318](https://github.com/QuantEcon/QuantEcon.py/pull/318) ([sglyon](https://github.com/sglyon))
- Fix math rendering in docstrings [\#315](https://github.com/QuantEcon/QuantEcon.py/pull/315) ([natashawatkins](https://github.com/natashawatkins))
- FIX: support\_enumeration: Use `\_numba\_linalg\_solve` [\#311](https://github.com/QuantEcon/QuantEcon.py/pull/311) ([oyamad](https://github.com/oyamad))

## [0.3.5.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.5.1) (2017-05-17)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.5...0.3.5.1)

**Merged pull requests:**

- remove dependency on statsmodels and use matrices to run OLS [\#310](https://github.com/QuantEcon/QuantEcon.py/pull/310) ([natashawatkins](https://github.com/natashawatkins))

## [0.3.5](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.5) (2017-05-16)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.4...0.3.5)

**Implemented enhancements:**

- Improved Change Log [\#243](https://github.com/QuantEcon/QuantEcon.py/issues/243)
- conda package [\#160](https://github.com/QuantEcon/QuantEcon.py/issues/160)

**Fixed bugs:**

- fix compatibility with pandas v0.2 [\#308](https://github.com/QuantEcon/QuantEcon.py/pull/308) ([natashawatkins](https://github.com/natashawatkins))

**Closed issues:**

- TEST: add test for compute\_fp.py [\#300](https://github.com/QuantEcon/QuantEcon.py/issues/300)
- Fix doc warnings [\#287](https://github.com/QuantEcon/QuantEcon.py/issues/287)
- Issue new release to PyPI and Conda-Forge [\#274](https://github.com/QuantEcon/QuantEcon.py/issues/274)
- Improvements in Test Coverage [\#254](https://github.com/QuantEcon/QuantEcon.py/issues/254)
- Test needed for `kalman` [\#200](https://github.com/QuantEcon/QuantEcon.py/issues/200)

**Merged pull requests:**

- TEST: Add tests for compute\_fp.py [\#309](https://github.com/QuantEcon/QuantEcon.py/pull/309) ([oyamad](https://github.com/oyamad))
- fix readthedocs badge [\#306](https://github.com/QuantEcon/QuantEcon.py/pull/306) ([mmcky](https://github.com/mmcky))
- Add a Gitter chat badge to README.md [\#305](https://github.com/QuantEcon/QuantEcon.py/pull/305) ([gitter-badger](https://github.com/gitter-badger))
- adjust coveragerc to skip @jit functions [\#303](https://github.com/QuantEcon/QuantEcon.py/pull/303) ([mmcky](https://github.com/mmcky))
- implement changes from PR \#157 to fix lqcontrol docstring [\#297](https://github.com/QuantEcon/QuantEcon.py/pull/297) ([mmcky](https://github.com/mmcky))
- DOC: Small correction [\#296](https://github.com/QuantEcon/QuantEcon.py/pull/296) ([oyamad](https://github.com/oyamad))
- WIP: fix doc warnings [\#292](https://github.com/QuantEcon/QuantEcon.py/pull/292) ([shizejin](https://github.com/shizejin))
- DOC: Edit rtd files [\#291](https://github.com/QuantEcon/QuantEcon.py/pull/291) ([oyamad](https://github.com/oyamad))
- remove specific version of quantecon from rtd environment [\#289](https://github.com/QuantEcon/QuantEcon.py/pull/289) ([mmcky](https://github.com/mmcky))
- DOC: Remove descriptions on `models` [\#288](https://github.com/QuantEcon/QuantEcon.py/pull/288) ([oyamad](https://github.com/oyamad))
- DOC: Generate doc files for game\_theory [\#286](https://github.com/QuantEcon/QuantEcon.py/pull/286) ([oyamad](https://github.com/oyamad))
- Add rouwenhorst method for approx AR\(1\) with MC [\#282](https://github.com/QuantEcon/QuantEcon.py/pull/282) ([sglyon](https://github.com/sglyon))

## [0.3.4](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.4) (2017-02-23)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.3...0.3.4)

**Implemented enhancements:**

- Python 3: The models/jv.py optimization [\#56](https://github.com/QuantEcon/QuantEcon.py/issues/56)
- Test: asset\_pricing [\#55](https://github.com/QuantEcon/QuantEcon.py/issues/55)
- Test: robustlq [\#54](https://github.com/QuantEcon/QuantEcon.py/issues/54)
- Test: lss [\#53](https://github.com/QuantEcon/QuantEcon.py/issues/53)
- Test: lae [\#52](https://github.com/QuantEcon/QuantEcon.py/issues/52)
- Test: estspec [\#51](https://github.com/QuantEcon/QuantEcon.py/issues/51)
- Profiling [\#44](https://github.com/QuantEcon/QuantEcon.py/issues/44)

**Fixed bugs:**

- Python 3: The models/jv.py optimization [\#56](https://github.com/QuantEcon/QuantEcon.py/issues/56)

**Closed issues:**

- ValueError: Unsupported target: parallel [\#275](https://github.com/QuantEcon/QuantEcon.py/issues/275)
- Updates to numba broke old code [\#269](https://github.com/QuantEcon/QuantEcon.py/issues/269)
- game\_theory: Add brute force pure nash [\#266](https://github.com/QuantEcon/QuantEcon.py/issues/266)
- No Conda recipe [\#259](https://github.com/QuantEcon/QuantEcon.py/issues/259)
- qe.distributions -- A Python version of Julia's Distributions.jl package [\#257](https://github.com/QuantEcon/QuantEcon.py/issues/257)

**Merged pull requests:**

- BUG: Change floats to ints in \_qnwsimp1 [\#284](https://github.com/QuantEcon/QuantEcon.py/pull/284) ([oyamad](https://github.com/oyamad))
- FIX: support\_enumeration: Disable cache in \_indiff\_mixed\_action [\#283](https://github.com/QuantEcon/QuantEcon.py/pull/283) ([oyamad](https://github.com/oyamad))
- upgraded tic, tac and toc to top level [\#280](https://github.com/QuantEcon/QuantEcon.py/pull/280) ([jstac](https://github.com/jstac))
- improve dependency fetcher with clearer parameters [\#279](https://github.com/QuantEcon/QuantEcon.py/pull/279) ([mmcky](https://github.com/mmcky))
- TRAVIS: Remove `=0.28.1` from conda install numba [\#277](https://github.com/QuantEcon/QuantEcon.py/pull/277) ([oyamad](https://github.com/oyamad))
- Add brute force method for finding pure nash equilibria. [\#276](https://github.com/QuantEcon/QuantEcon.py/pull/276) ([shizejin](https://github.com/shizejin))
- Implement the "imitation game algorithm" by McLennan-Tourky [\#273](https://github.com/QuantEcon/QuantEcon.py/pull/273) ([oyamad](https://github.com/oyamad))
- game\_theory: Add random game generators [\#270](https://github.com/QuantEcon/QuantEcon.py/pull/270) ([oyamad](https://github.com/oyamad))
- game\_theory: Add lemke\_howson [\#268](https://github.com/QuantEcon/QuantEcon.py/pull/268) ([oyamad](https://github.com/oyamad))
- Player, NormalFormGame: Make `payoff\_array`'s C contiguous [\#265](https://github.com/QuantEcon/QuantEcon.py/pull/265) ([oyamad](https://github.com/oyamad))
- Change Miniconda-latest to Miniconda3-latest [\#264](https://github.com/QuantEcon/QuantEcon.py/pull/264) ([oyamad](https://github.com/oyamad))
- game\_theory: Add support\_enumeration [\#263](https://github.com/QuantEcon/QuantEcon.py/pull/263) ([oyamad](https://github.com/oyamad))
- Remove warnings supression as noted by Issue \#229 [\#231](https://github.com/QuantEcon/QuantEcon.py/pull/231) ([mmcky](https://github.com/mmcky))

## [0.3.3](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.3) (2016-07-21)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.2...0.3.3)

**Closed issues:**

- Release new PyPI version and update the change log [\#249](https://github.com/QuantEcon/QuantEcon.py/issues/249)

**Merged pull requests:**

- Update to version 0.3.3 [\#260](https://github.com/QuantEcon/QuantEcon.py/pull/260) ([mmcky](https://github.com/mmcky))
- ddp: Export sa\_indices [\#255](https://github.com/QuantEcon/QuantEcon.py/pull/255) ([oyamad](https://github.com/oyamad))
- probvec: Use guvectorize with target='parallel' [\#253](https://github.com/QuantEcon/QuantEcon.py/pull/253) ([oyamad](https://github.com/oyamad))

## [0.3.2](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.2) (2016-04-25)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3.1...0.3.2)

**Implemented enhancements:**

- TODO: Add states/nodes to MarkovChain/DiGraph [\#101](https://github.com/QuantEcon/QuantEcon.py/issues/101)

**Fixed bugs:**

- readthedocs - missing a few autojit functions [\#170](https://github.com/QuantEcon/QuantEcon.py/issues/170)

**Closed issues:**

- TODO: DiscreteDP: Allow beta=1 [\#242](https://github.com/QuantEcon/QuantEcon.py/issues/242)
- \[Documentation\] Update qe\_api.py to incorporate a nicer title for Game Theory Module [\#239](https://github.com/QuantEcon/QuantEcon.py/issues/239)
- filterwarnings [\#229](https://github.com/QuantEcon/QuantEcon.py/issues/229)
- Generate new release v0.3.1 for PyPI [\#225](https://github.com/QuantEcon/QuantEcon.py/issues/225)
- `requests` required [\#223](https://github.com/QuantEcon/QuantEcon.py/issues/223)
- Solution to Schelling example does not correspond to model description [\#212](https://github.com/QuantEcon/QuantEcon.py/issues/212)
- Upgrade of graphs to Matplotlib 1.5.0 [\#209](https://github.com/QuantEcon/QuantEcon.py/issues/209)
- Aiyagari examples [\#203](https://github.com/QuantEcon/QuantEcon.py/issues/203)
- Improve build times for Travis by removing sudo commands from .travis.yml [\#167](https://github.com/QuantEcon/QuantEcon.py/issues/167)
- Python wheels [\#141](https://github.com/QuantEcon/QuantEcon.py/issues/141)

**Merged pull requests:**

- MarkovChain: Bug fix in \_compute\_stationary with state\_values [\#252](https://github.com/QuantEcon/QuantEcon.py/pull/252) ([oyamad](https://github.com/oyamad))
- Remove code that supported optional numba installation [\#251](https://github.com/QuantEcon/QuantEcon.py/pull/251) ([mmcky](https://github.com/mmcky))
- Modified tauchen to return a MarkovChain instance that stores both states and transitions [\#250](https://github.com/QuantEcon/QuantEcon.py/pull/250) ([jstac](https://github.com/jstac))
- DiscreteDP: Allow beta=1 [\#244](https://github.com/QuantEcon/QuantEcon.py/pull/244) ([oyamad](https://github.com/oyamad))
- Update name of Game Theory for cleaner look [\#241](https://github.com/QuantEcon/QuantEcon.py/pull/241) ([mmcky](https://github.com/mmcky))
- Add states/nodes to MarkovChain/DiGraph [\#237](https://github.com/QuantEcon/QuantEcon.py/pull/237) ([oyamad](https://github.com/oyamad))
- Drop `num\_actions` from DiscreteDP [\#236](https://github.com/QuantEcon/QuantEcon.py/pull/236) ([oyamad](https://github.com/oyamad))
- Update rtd to a conda environment and remove mock to correct for missing `jit` functions [\#235](https://github.com/QuantEcon/QuantEcon.py/pull/235) ([mmcky](https://github.com/mmcky))
- Improve nb fetch utility so that it doesn't overwrite files by default [\#233](https://github.com/QuantEcon/QuantEcon.py/pull/233) ([mmcky](https://github.com/mmcky))
- Updating requirements file [\#232](https://github.com/QuantEcon/QuantEcon.py/pull/232) ([mmcky](https://github.com/mmcky))
- Adjustments to cleanup travis CI [\#230](https://github.com/QuantEcon/QuantEcon.py/pull/230) ([mmcky](https://github.com/mmcky))
- Sl/numba tauchen [\#227](https://github.com/QuantEcon/QuantEcon.py/pull/227) ([sglyon](https://github.com/sglyon))
- Changes in normal\_form\_game [\#226](https://github.com/QuantEcon/QuantEcon.py/pull/226) ([oyamad](https://github.com/oyamad))

## [0.3.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.3.1) (2016-01-22)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.3...0.3.1)

**Merged pull requests:**

- Add `distributions` to quantecon/\_\_init\_\_.py [\#224](https://github.com/QuantEcon/QuantEcon.py/pull/224) ([oyamad](https://github.com/oyamad))
- Change converter functions to use default float [\#222](https://github.com/QuantEcon/QuantEcon.py/pull/222) ([cc7768](https://github.com/cc7768))
- Update base api to include module and object imports, fix missing imp… [\#221](https://github.com/QuantEcon/QuantEcon.py/pull/221) ([mmcky](https://github.com/mmcky))
- Add game\_theory.normal\_form\_game [\#220](https://github.com/QuantEcon/QuantEcon.py/pull/220) ([oyamad](https://github.com/oyamad))
- Update docs to remove models subpackage and update the qe\_api.py scri… [\#219](https://github.com/QuantEcon/QuantEcon.py/pull/219) ([mmcky](https://github.com/mmcky))

## [0.3](https://github.com/QuantEcon/QuantEcon.py/tree/0.3) (2016-01-07)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/pre-migrate-applications...0.3)

**Closed issues:**

- Importing without Numba [\#214](https://github.com/QuantEcon/QuantEcon.py/issues/214)
- Relocate models subpackage  [\#205](https://github.com/QuantEcon/QuantEcon.py/issues/205)
- Python 3.4 Review and Default ... [\#192](https://github.com/QuantEcon/QuantEcon.py/issues/192)
- NumbaWarning with Python 3 [\#155](https://github.com/QuantEcon/QuantEcon.py/issues/155)

**Merged pull requests:**

- Add notebook autosetup and fetch utility for notebooks ... [\#217](https://github.com/QuantEcon/QuantEcon.py/pull/217) ([mmcky](https://github.com/mmcky))
- added asarray in discrete\_rv [\#215](https://github.com/QuantEcon/QuantEcon.py/pull/215) ([jstac](https://github.com/jstac))
- Migrate applications .. examples/ and quantecon/models/ to QuantEcon.applications [\#211](https://github.com/QuantEcon/QuantEcon.py/pull/211) ([mmcky](https://github.com/mmcky))

## [pre-migrate-applications](https://github.com/QuantEcon/QuantEcon.py/tree/pre-migrate-applications) (2015-11-23)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.2.2...pre-migrate-applications)

**Fixed bugs:**

- Broken examples python files [\#179](https://github.com/QuantEcon/QuantEcon.py/issues/179)

**Closed issues:**

- \[Dependancy\] Propose introduction of dependancy on Numba [\#173](https://github.com/QuantEcon/QuantEcon.py/issues/173)

**Merged pull requests:**

- Adjust Travis-CI to run in python 3.5 and adjust pip classifiers [\#210](https://github.com/QuantEcon/QuantEcon.py/pull/210) ([mmcky](https://github.com/mmcky))
- BUG: Fix bug in probvec [\#208](https://github.com/QuantEcon/QuantEcon.py/pull/208) ([oyamad](https://github.com/oyamad))
- TRAVIS: set destination path for miniconda [\#207](https://github.com/QuantEcon/QuantEcon.py/pull/207) ([oyamad](https://github.com/oyamad))
- DiscreteDP: further fix in docstring [\#204](https://github.com/QuantEcon/QuantEcon.py/pull/204) ([oyamad](https://github.com/oyamad))
- Whitener [\#202](https://github.com/QuantEcon/QuantEcon.py/pull/202) ([thomassargent30](https://github.com/thomassargent30))
- Fix optgrowth solution notebook [\#188](https://github.com/QuantEcon/QuantEcon.py/pull/188) ([oyamad](https://github.com/oyamad))

## [0.2.2](https://github.com/QuantEcon/QuantEcon.py/tree/0.2.2) (2015-10-06)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.2.1...0.2.2)

## [0.2.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.2.1) (2015-10-05)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.10...0.2.1)

**Implemented enhancements:**

- Possible enhancement for DiscreteDP class [\#180](https://github.com/QuantEcon/QuantEcon.py/issues/180)

**Closed issues:**

- Arellano\_solutions migration to matplotlib 1.4.3 in python 3.4 [\#190](https://github.com/QuantEcon/QuantEcon.py/issues/190)

**Merged pull requests:**

- Fix P update in stationary coefficients. [\#199](https://github.com/QuantEcon/QuantEcon.py/pull/199) ([thomassargent30](https://github.com/thomassargent30))
- DiscreteDDP: minor fix in docstring and pep8 compliance [\#198](https://github.com/QuantEcon/QuantEcon.py/pull/198) ([oyamad](https://github.com/oyamad))
- sample\_without\_replacement refactored [\#196](https://github.com/QuantEcon/QuantEcon.py/pull/196) ([oyamad](https://github.com/oyamad))
- Fix py34 [\#194](https://github.com/QuantEcon/QuantEcon.py/pull/194) ([mmcky](https://github.com/mmcky))
- Fix: MarkovChain.simulate [\#193](https://github.com/QuantEcon/QuantEcon.py/pull/193) ([oyamad](https://github.com/oyamad))
- Update for lakemodel\_solutions.ipynb [\#191](https://github.com/QuantEcon/QuantEcon.py/pull/191) ([mmcky](https://github.com/mmcky))
- Revise DiscreteDP solution notebook [\#187](https://github.com/QuantEcon/QuantEcon.py/pull/187) ([oyamad](https://github.com/oyamad))
- Fix readthedocs [\#186](https://github.com/QuantEcon/QuantEcon.py/pull/186) ([mmcky](https://github.com/mmcky))
- probvec: Use less memory [\#184](https://github.com/QuantEcon/QuantEcon.py/pull/184) ([oyamad](https://github.com/oyamad))
- Fixes for examples and solutions notebook [\#183](https://github.com/QuantEcon/QuantEcon.py/pull/183) ([mmcky](https://github.com/mmcky))
- Testing infrastructure for examples and solutions notebook [\#182](https://github.com/QuantEcon/QuantEcon.py/pull/182) ([mmcky](https://github.com/mmcky))
- Add \_\_dir\_\_ to DPSolveResult [\#181](https://github.com/QuantEcon/QuantEcon.py/pull/181) ([oyamad](https://github.com/oyamad))
- DiscreteDP refactoring [\#177](https://github.com/QuantEcon/QuantEcon.py/pull/177) ([oyamad](https://github.com/oyamad))
- Removing unecessary python egg information [\#176](https://github.com/QuantEcon/QuantEcon.py/pull/176) ([mmcky](https://github.com/mmcky))
- MarkovChain: Sparse matrix support [\#174](https://github.com/QuantEcon/QuantEcon.py/pull/174) ([oyamad](https://github.com/oyamad))
- MDP [\#171](https://github.com/QuantEcon/QuantEcon.py/pull/171) ([oyamad](https://github.com/oyamad))

## [0.1.10](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.10) (2015-08-28)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.9...0.1.10)

**Implemented enhancements:**

- Update MarkovChain with ``replicate `` method and Future Numba Improvements [\#146](https://github.com/QuantEcon/QuantEcon.py/issues/146)
- Sparse Matrix Implementations [\#145](https://github.com/QuantEcon/QuantEcon.py/issues/145)

**Fixed bugs:**

- Bug in robust\_monopolist.py [\#92](https://github.com/QuantEcon/QuantEcon.py/issues/92)
- Python3.4 Compatibility Issues ... Package Support [\#61](https://github.com/QuantEcon/QuantEcon.py/issues/61)

**Closed issues:**

- Migrate some modules to util subpackage [\#165](https://github.com/QuantEcon/QuantEcon.py/issues/165)
- mc\_sample\_path needs a test [\#148](https://github.com/QuantEcon/QuantEcon.py/issues/148)
- readthedocs broken [\#138](https://github.com/QuantEcon/QuantEcon.py/issues/138)
- Numba version of mc\_sample\_path [\#137](https://github.com/QuantEcon/QuantEcon.py/issues/137)
- Numba warning --- implement a common warning [\#133](https://github.com/QuantEcon/QuantEcon.py/issues/133)
- Numpy version to fall back on in lss.py [\#132](https://github.com/QuantEcon/QuantEcon.py/issues/132)
- Incorrect instructions for dev environment on QuantEcon wiki? [\#126](https://github.com/QuantEcon/QuantEcon.py/issues/126)
- Possibility of implementing an improved FRB/US model within Python using QuantEcon? [\#121](https://github.com/QuantEcon/QuantEcon.py/issues/121)
- Accessing the cartesian routine [\#98](https://github.com/QuantEcon/QuantEcon.py/issues/98)
- Broken links in docstrings / readthedocs, slightly urgent [\#91](https://github.com/QuantEcon/QuantEcon.py/issues/91)
- Performance [\#36](https://github.com/QuantEcon/QuantEcon.py/issues/36)
- Wiki Testing ... Latex Matrix Support? [\#30](https://github.com/QuantEcon/QuantEcon.py/issues/30)
- Document Extended Testing Data Requirements [\#27](https://github.com/QuantEcon/QuantEcon.py/issues/27)

**Merged pull requests:**

- Update util subpackage to include additional utilities [\#172](https://github.com/QuantEcon/QuantEcon.py/pull/172) ([mmcky](https://github.com/mmcky))
- STY: pep8ified kalman.py [\#168](https://github.com/QuantEcon/QuantEcon.py/pull/168) ([sglyon](https://github.com/sglyon))
- MarkovChain.simulate API change; random\_state option added [\#166](https://github.com/QuantEcon/QuantEcon.py/pull/166) ([oyamad](https://github.com/oyamad))
- Update readthedocs to include all files in quantecon [\#164](https://github.com/QuantEcon/QuantEcon.py/pull/164) ([mmcky](https://github.com/mmcky))
- fix broken links for Issue \#91 [\#163](https://github.com/QuantEcon/QuantEcon.py/pull/163) ([mmcky](https://github.com/mmcky))
- Add the same test case as in the Julia version [\#161](https://github.com/QuantEcon/QuantEcon.py/pull/161) ([oyamad](https://github.com/oyamad))
- Random MarkovChain [\#154](https://github.com/QuantEcon/QuantEcon.py/pull/154) ([oyamad](https://github.com/oyamad))
- Update warning message if numba import fails [\#151](https://github.com/QuantEcon/QuantEcon.py/pull/151) ([mmcky](https://github.com/mmcky))
- Adjust mock and environment requirements for proper compilation on RTD [\#150](https://github.com/QuantEcon/QuantEcon.py/pull/150) ([mmcky](https://github.com/mmcky))
- REF: renamed cartesian.py as gridtools.py [\#149](https://github.com/QuantEcon/QuantEcon.py/pull/149) ([albop](https://github.com/albop))
- Numba improvements [\#144](https://github.com/QuantEcon/QuantEcon.py/pull/144) ([sglyon](https://github.com/sglyon))

## [0.1.9](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.9) (2015-04-17)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.8...0.1.9)

**Closed issues:**

- Need a MANIFEST.in file? [\#135](https://github.com/QuantEcon/QuantEcon.py/issues/135)

**Merged pull requests:**

- Adding Manifest In File to Distribute LICENSE and README.md [\#136](https://github.com/QuantEcon/QuantEcon.py/pull/136) ([mmcky](https://github.com/mmcky))
- Update kalman [\#134](https://github.com/QuantEcon/QuantEcon.py/pull/134) ([jstac](https://github.com/jstac))

## [0.1.8](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.8) (2015-04-07)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.7...0.1.8)

**Closed issues:**

- quantecon.models.\_\_init\_\_.py problem [\#123](https://github.com/QuantEcon/QuantEcon.py/issues/123)

**Merged pull requests:**

- numba-fied simulation [\#131](https://github.com/QuantEcon/QuantEcon.py/pull/131) ([jstac](https://github.com/jstac))
- CC: Added Balint and Tom's oligopoly.py edits. [\#130](https://github.com/QuantEcon/QuantEcon.py/pull/130) ([cc7768](https://github.com/cc7768))
- Update lss [\#129](https://github.com/QuantEcon/QuantEcon.py/pull/129) ([jstac](https://github.com/jstac))
- ENH: added formatting to iteration printing [\#124](https://github.com/QuantEcon/QuantEcon.py/pull/124) ([sglyon](https://github.com/sglyon))
- Fixed imports for solow module...again! [\#122](https://github.com/QuantEcon/QuantEcon.py/pull/122) ([davidrpugh](https://github.com/davidrpugh))
- Fixed import statements in \_\_init\_\_.py to include solow module. [\#120](https://github.com/QuantEcon/QuantEcon.py/pull/120) ([davidrpugh](https://github.com/davidrpugh))
- Convert longdescription from markdown to rst [\#119](https://github.com/QuantEcon/QuantEcon.py/pull/119) ([mmcky](https://github.com/mmcky))

## [0.1.7](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.7) (2015-02-09)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.6...0.1.7)

**Implemented enhancements:**

- Modify LQ class in lqcontrol.py so it can handle cross product terms [\#87](https://github.com/QuantEcon/QuantEcon.py/issues/87)

**Fixed bugs:**

- \[Examples\] mc\_convergence\_plot.py ``IndexError`` [\#93](https://github.com/QuantEcon/QuantEcon.py/issues/93)

**Closed issues:**

- pylab [\#95](https://github.com/QuantEcon/QuantEcon.py/issues/95)
- Rougier-Müller-Varoquaux link in Matplotlib section [\#88](https://github.com/QuantEcon/QuantEcon.py/issues/88)
- Improve PyPi quantecon page [\#85](https://github.com/QuantEcon/QuantEcon.py/issues/85)
- display  methods [\#13](https://github.com/QuantEcon/QuantEcon.py/issues/13)
- Testing [\#10](https://github.com/QuantEcon/QuantEcon.py/issues/10)

**Merged pull requests:**

- Move Solow model notebook to website repo. [\#116](https://github.com/QuantEcon/QuantEcon.py/pull/116) ([davidrpugh](https://github.com/davidrpugh))
- ENH: adding \_\_str\_\_ and \_\_repr\_\_ methods to classes [\#114](https://github.com/QuantEcon/QuantEcon.py/pull/114) ([sglyon](https://github.com/sglyon))
- BUG: Found bug in solutions notebook  [\#113](https://github.com/QuantEcon/QuantEcon.py/pull/113) ([sglyon](https://github.com/sglyon))
- Adding more information for the PYPI README [\#112](https://github.com/QuantEcon/QuantEcon.py/pull/112) ([mmcky](https://github.com/mmcky))
- STY: made all .py files pep8 compliant [\#110](https://github.com/QuantEcon/QuantEcon.py/pull/110) ([sglyon](https://github.com/sglyon))
- FIX: Discrete Lyapunov with complex input [\#108](https://github.com/QuantEcon/QuantEcon.py/pull/108) ([ChadFulton](https://github.com/ChadFulton))
- Update optgrowth.py [\#106](https://github.com/QuantEcon/QuantEcon.py/pull/106) ([akshayshanker](https://github.com/akshayshanker))
- Updating Developer and Coordinator links to website not github readme [\#105](https://github.com/QuantEcon/QuantEcon.py/pull/105) ([mmcky](https://github.com/mmcky))
- Consolidating duplicate information and adding links to QuantEcon.site README [\#104](https://github.com/QuantEcon/QuantEcon.py/pull/104) ([mmcky](https://github.com/mmcky))
- BUG: Fix bug for input with int elements [\#102](https://github.com/QuantEcon/QuantEcon.py/pull/102) ([oyamad](https://github.com/oyamad))
- Refactor gth\_solve.py; add documentation of gth\_solve and graph\_tools [\#100](https://github.com/QuantEcon/QuantEcon.py/pull/100) ([oyamad](https://github.com/oyamad))
- BUG: cartesian grids also work with integers [\#99](https://github.com/QuantEcon/QuantEcon.py/pull/99) ([albop](https://github.com/albop))
- Fixing links at the top of solutions/ ipython notebooks to new quant-econ layout [\#96](https://github.com/QuantEcon/QuantEcon.py/pull/96) ([sanguineturtle](https://github.com/sanguineturtle))
- Fixes for Broken Examples ...  [\#94](https://github.com/QuantEcon/QuantEcon.py/pull/94) ([sanguineturtle](https://github.com/sanguineturtle))
- Add solow model [\#74](https://github.com/QuantEcon/QuantEcon.py/pull/74) ([davidrpugh](https://github.com/davidrpugh))

## [0.1.6](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.6) (2014-11-04)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.5...0.1.6)

**Implemented enhancements:**

- Add Version Attribute ... [\#62](https://github.com/QuantEcon/QuantEcon.py/issues/62)
- Improve website [\#12](https://github.com/QuantEcon/QuantEcon.py/issues/12)

**Fixed bugs:**

- BUG: there is a bug in the tests for lss [\#65](https://github.com/QuantEcon/QuantEcon.py/issues/65)

**Closed issues:**

- Attribute lists in docs [\#89](https://github.com/QuantEcon/QuantEcon.py/issues/89)
- kalman.py: column vector or row vector? [\#81](https://github.com/QuantEcon/QuantEcon.py/issues/81)
- Testing in a Conda development environment [\#80](https://github.com/QuantEcon/QuantEcon.py/issues/80)
- Accessing unset `f\_args` and `jac\_args` attributes of `ivp.IVP` class should return `None` [\#77](https://github.com/QuantEcon/QuantEcon.py/issues/77)
- Contributing to quantecon via git submodules [\#72](https://github.com/QuantEcon/QuantEcon.py/issues/72)
- CLN: move plotting functions out of library [\#70](https://github.com/QuantEcon/QuantEcon.py/issues/70)
- LLN and CLT [\#67](https://github.com/QuantEcon/QuantEcon.py/issues/67)
- Linear Quadratic Nash [\#64](https://github.com/QuantEcon/QuantEcon.py/issues/64)
- bellman to bellman\_operator in career.py  [\#59](https://github.com/QuantEcon/QuantEcon.py/issues/59)
- Readthedocs URL [\#34](https://github.com/QuantEcon/QuantEcon.py/issues/34)
- Using state of the art C and Fortran libraries in quantecon [\#26](https://github.com/QuantEcon/QuantEcon.py/issues/26)
- Starting work on an IVP solver... [\#22](https://github.com/QuantEcon/QuantEcon.py/issues/22)

**Merged pull requests:**

- MARKOV: Minor corrections [\#86](https://github.com/QuantEcon/QuantEcon.py/pull/86) ([oyamad](https://github.com/oyamad))
- ENH: added tic, tac, toc functions [\#83](https://github.com/QuantEcon/QuantEcon.py/pull/83) ([albop](https://github.com/albop))
- Rtfd: Bring in some of the working files. [\#82](https://github.com/QuantEcon/QuantEcon.py/pull/82) ([cc7768](https://github.com/cc7768))
- MARKOV: More efficient implementation for computing stationary distributions [\#79](https://github.com/QuantEcon/QuantEcon.py/pull/79) ([oyamad](https://github.com/oyamad))
- Closes issue \#77. [\#78](https://github.com/QuantEcon/QuantEcon.py/pull/78) ([davidrpugh](https://github.com/davidrpugh))
- Hotfix for slight notebook issues raised by @jstac. [\#73](https://github.com/QuantEcon/QuantEcon.py/pull/73) ([davidrpugh](https://github.com/davidrpugh))
- Retry add ivp solver [\#71](https://github.com/QuantEcon/QuantEcon.py/pull/71) ([davidrpugh](https://github.com/davidrpugh))
- Lqnash [\#69](https://github.com/QuantEcon/QuantEcon.py/pull/69) ([cc7768](https://github.com/cc7768))
- Fixed indexing in illustrates\_lln.py [\#68](https://github.com/QuantEcon/QuantEcon.py/pull/68) ([oyamad](https://github.com/oyamad))
- Lss bug [\#66](https://github.com/QuantEcon/QuantEcon.py/pull/66) ([cc7768](https://github.com/cc7768))
- Adding Version Attribute to QuantEcon class [\#63](https://github.com/QuantEcon/QuantEcon.py/pull/63) ([sanguineturtle](https://github.com/sanguineturtle))
- CAREER: Changed all occurences of bellman to bellman\_operator for unifie... [\#60](https://github.com/QuantEcon/QuantEcon.py/pull/60) ([cc7768](https://github.com/cc7768))

## [0.1.5](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.5) (2014-08-20)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.4...0.1.5)

**Closed issues:**

- Solving Lyapunov equations [\#42](https://github.com/QuantEcon/QuantEcon.py/issues/42)

**Merged pull requests:**

- Update linproc.py [\#57](https://github.com/QuantEcon/QuantEcon.py/pull/57) ([akshayshanker](https://github.com/akshayshanker))
- TRAVIS: Small changes so that tests pass. [\#49](https://github.com/QuantEcon/QuantEcon.py/pull/49) ([cc7768](https://github.com/cc7768))
- Lyapunov [\#48](https://github.com/QuantEcon/QuantEcon.py/pull/48) ([cc7768](https://github.com/cc7768))

## [0.1.4](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.4) (2014-08-10)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.3...0.1.4)

**Implemented enhancements:**

- Shift some functions / classes / modules to 'models' subpackage [\#37](https://github.com/QuantEcon/QuantEcon.py/issues/37)
- Python 3 compatibility [\#16](https://github.com/QuantEcon/QuantEcon.py/issues/16)

**Fixed bugs:**

- nosetests issue with /test-script/ in the base level script [\#28](https://github.com/QuantEcon/QuantEcon.py/issues/28)
- Negative values obtained for stationary distribution in mc\_tools.py [\#18](https://github.com/QuantEcon/QuantEcon.py/issues/18)

**Closed issues:**

- tests module missing \_\_init\_\_.py [\#29](https://github.com/QuantEcon/QuantEcon.py/issues/29)
- Dependency list for quantecon? [\#20](https://github.com/QuantEcon/QuantEcon.py/issues/20)
- Computing Markov Stationary Distributions in mc\_tools.py [\#19](https://github.com/QuantEcon/QuantEcon.py/issues/19)
- Documentation [\#15](https://github.com/QuantEcon/QuantEcon.py/issues/15)
- 3 contiguous 46 bit pieces of memory? [\#2](https://github.com/QuantEcon/QuantEcon.py/issues/2)

**Merged pull requests:**

- Tests [\#45](https://github.com/QuantEcon/QuantEcon.py/pull/45) ([cc7768](https://github.com/cc7768))
- MARKOV: Put the Markov functions into a class wrapper and created a way ... [\#43](https://github.com/QuantEcon/QuantEcon.py/pull/43) ([cc7768](https://github.com/cc7768))
- ENH: moved the lucas\_tree tuple to a new LucasTree class [\#41](https://github.com/QuantEcon/QuantEcon.py/pull/41) ([sglyon](https://github.com/sglyon))
- Model subpackage [\#38](https://github.com/QuantEcon/QuantEcon.py/pull/38) ([sglyon](https://github.com/sglyon))
- Compat [\#33](https://github.com/QuantEcon/QuantEcon.py/pull/33) ([cc7768](https://github.com/cc7768))
- DOCS: Added a pip-requirements so that readthedocs can build the documentation [\#25](https://github.com/QuantEcon/QuantEcon.py/pull/25) ([cc7768](https://github.com/cc7768))
- Docs [\#24](https://github.com/QuantEcon/QuantEcon.py/pull/24) ([cc7768](https://github.com/cc7768))
- Quadrature routines [\#17](https://github.com/QuantEcon/QuantEcon.py/pull/17) ([sglyon](https://github.com/sglyon))
- Basic setup for housing tests ... [\#11](https://github.com/QuantEcon/QuantEcon.py/pull/11) ([sanguineturtle](https://github.com/sanguineturtle))
- BUG: fixed bug in asset pricing solutions [\#9](https://github.com/QuantEcon/QuantEcon.py/pull/9) ([sglyon](https://github.com/sglyon))
- BUG: numpy broadcasting error in asset pricing [\#8](https://github.com/QuantEcon/QuantEcon.py/pull/8) ([sglyon](https://github.com/sglyon))
- ENH: don't solve linear system twice in a row [\#7](https://github.com/QuantEcon/QuantEcon.py/pull/7) ([sglyon](https://github.com/sglyon))
- Update and Delete solutions/stand\_alone\_programs/ folder [\#6](https://github.com/QuantEcon/QuantEcon.py/pull/6) ([sanguineturtle](https://github.com/sanguineturtle))
- Updates to examples/ and solutions/stand\_alone\_programs to use quantecon as a package [\#5](https://github.com/QuantEcon/QuantEcon.py/pull/5) ([sanguineturtle](https://github.com/sanguineturtle))

## [0.1.3](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.3) (2014-06-10)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.2...0.1.3)

**Merged pull requests:**

- Changed build dependencies to reduced requirements and re-added run dependencies  [\#4](https://github.com/QuantEcon/QuantEcon.py/pull/4) ([sanguineturtle](https://github.com/sanguineturtle))

## [0.1.2](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.2) (2014-05-26)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1.1...0.1.2)

**Merged pull requests:**

- Added \_\_init\_\_.py with import statements and updated .gitignore [\#3](https://github.com/QuantEcon/QuantEcon.py/pull/3) ([sanguineturtle](https://github.com/sanguineturtle))

## [0.1.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.1.1) (2014-05-21)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.py/compare/0.1...0.1.1)

## [0.1](https://github.com/QuantEcon/QuantEcon.py/tree/0.1) (2014-05-21)


\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*