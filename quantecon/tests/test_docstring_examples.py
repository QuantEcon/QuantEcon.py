"""
Execute every docstring example in the package, without comparing output.

Examples are executable specifications: if one stops running, a reader
who copies it gets an error. This sweep catches exactly that -- missing
imports, renamed functions, changed signatures -- and nothing else.

It deliberately does not compare an example's printed output with the
text in the docstring. That comparison is what `--doctest-modules`
performs, and it fails whenever NumPy changes an array repr or a float
format, independently of whether the example still works. Most of the
31 examples repaired in gh-864 had rotted that way (pre-NumPy-1.14
array spacing, NumPy 2 scalar reprs), so keeping the comparison would
commit us to re-pasting expected output after every NumPy release.
See gh-866 for the discussion.

"""
import contextlib
import doctest
import importlib
import io
import pkgutil

import numpy as np
import pytest

import quantecon


# `fetch_nb_dependencies` fetches over the network, so its example cannot
# run offline. The module is deprecated and is removed in v1.0 (gh-880),
# which retires this entry with it.
SKIP_MODULES = {'quantecon.util.notebooks'}


def _collect_doctests():
    """
    Return every docstring in the package that carries examples.

    """
    collected = []
    for module_info in pkgutil.walk_packages(quantecon.__path__, 'quantecon.'):
        name = module_info.name
        if '.tests' in name or name in SKIP_MODULES:
            continue
        module = importlib.import_module(name)
        for test in doctest.DocTestFinder().find(module):
            if test.examples:
                collected.append(test)
    return collected


DOCTESTS = _collect_doctests()


def test_doctests_are_collected():
    """
    Guard against the sweep silently becoming a no-op.

    Every test below is generated from `DOCTESTS`, so an empty
    collection would report success while checking nothing -- the same
    way these examples rotted in the first place.

    """
    assert len(DOCTESTS) > 40


@pytest.mark.parametrize('test', DOCTESTS, ids=lambda test: test.name)
def test_docstring_example_runs(test):
    """
    Execute one docstring's examples, discarding their output.

    The examples of a docstring share a namespace, as they do under
    `doctest`, so a later line may use names bound by an earlier one.

    """
    printoptions = np.get_printoptions()
    namespace = dict(test.globs)
    try:
        for example in test.examples:
            code = compile(example.source, '<%s>' % test.name, 'single')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, namespace)
            except Exception:
                # An example documenting a traceback is meant to raise.
                if example.exc_msg is None:
                    raise
    finally:
        # Several `game_theory` examples call `np.set_printoptions` for
        # readability and do not restore it; print options are global.
        np.set_printoptions(**printoptions)
