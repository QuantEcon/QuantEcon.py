
## Quantitative Economics (Python)

A code library for quantitative economic modeling in Python

Libary Website: [http://quantecon.org/python_index.html](http://quantecon.org/python_index.html)

### Installation

See the [library website](http://quantecon.org/python_index.html) for instructions

#### Build and Coverage Status:

[![Build Status](https://travis-ci.org/QuantEcon/QuantEcon.py.svg?branch=master)](https://travis-ci.org/QuantEcon/QuantEcon.py)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.png)](https://coveralls.io/r/QuantEcon/QuantEcon.py)

#### ReadTheDocs Status:

[![Documentation Status](https://readthedocs.org/projects/quanteconpy/badge/?version=latest)](http://quanteconpy.readthedocs.io/en/latest/?badge=latest)

#### Gitter

[![Join the chat at https://gitter.im/QuantEcon/QuantEcon.py](https://badges.gitter.im/QuantEcon/QuantEcon.py.svg)](https://gitter.im/QuantEcon/QuantEcon.py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


## Additional Links

1. [Project Coordinators](http://quantecon.org/about)
2. [Lead Developers](http://quantecon.org/about)
3. [QuantEcon Lecture Website](http://quant-econ.net)

### License

Copyright © 2013, 2014, 2015, 2016 Thomas J. Sargent and John Stachurski: BSD-3
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

### Ver. 0.3.4 (23-February-2017)
1. Add support_enumeration, a simple algorithm that computes all mixed-action Nash equilibria of a non-degenerate 2-player game. See [PR #263](https://github.com/QuantEcon/QuantEcon.py/pull/263)
2. Various fixes for issues with numba. See [PR #265](https://github.com/QuantEcon/QuantEcon.py/pull/265), [PR #283](https://github.com/QuantEcon/QuantEcon.py/pull/283)
3.Add lemke_howson algorithm to game_theory module. See [PR #268](https://github.com/QuantEcon/QuantEcon.py/pull/268)
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

