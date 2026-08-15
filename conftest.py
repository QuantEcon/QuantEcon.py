"""
Pytest configuration for the ``--doctest-modules`` run (see gh-866).

This lives at the repository root rather than inside the package on
purpose.  `docs/qe_apidoc.py` discovers the "Tools" pages by globbing
``../quantecon/[a-z0-9]*.py``, so a `quantecon/conftest.py` would be
picked up as a public module and generate a docs page for itself,
failing the docs-drift check in CI.  Keeping it here also keeps a
module that imports pytest -- not a runtime dependency -- out of the
installed wheel.

"""
import numpy as np
import pytest

# These modules have docstring examples that cannot pass verbatim:
# `fetch_nb_dependencies` fetches over the network, and the `timing`
# examples print wall-clock durations that vary per run.  Exclude them
# here rather than annotating the examples with `# doctest: +SKIP`,
# which would render visibly in the published docs.  Paths are relative
# to this file, i.e. to the repository root.
collect_ignore = [
    "quantecon/util/notebooks.py",
    "quantecon/util/timing.py",
]


@pytest.fixture(autouse=True)
def _restore_printoptions():
    """
    Restore NumPy print options after each test.

    Several `game_theory` docstring examples call
    `np.set_printoptions(precision=4)` for readability and deliberately
    do not restore it.  Print options are process-global, so without
    this fixture every doctest collected after those examples would
    render arrays at the leaked precision and fail.

    """
    saved = np.get_printoptions()
    yield
    np.set_printoptions(**saved)
