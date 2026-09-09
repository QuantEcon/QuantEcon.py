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
import linecache
import os
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


def _run_examples(test):
    """
    Execute the examples of one docstring, discarding their output.

    The examples share a namespace, as they do under `doctest`, so a
    later line may use names bound by an earlier one. An example that
    raises is reported by source file, line and position, with the
    original exception chained so a failure inside library code keeps
    its full traceback.

    """
    __tracebackhide__ = True
    namespace = dict(test.globs)
    for i, example in enumerate(test.examples, 1):
        filename = '<doctest %s[%d]>' % (test.name, i)
        # Register the source so a traceback shows the example's line
        # instead of '???'; doctest's own runner does the same.
        linecache.cache[filename] = (
            len(example.source), None,
            example.source.splitlines(keepends=True), filename)
        code = compile(example.source, filename, 'single')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)
        except Exception as exc:
            # An example documenting a traceback is meant to raise.
            if example.exc_msg is not None:
                continue
            # Both line numbers are zero-based: the docstring's within
            # the file, and the example's within the docstring.
            lineno = test.lineno + example.lineno + 1
            raise AssertionError(
                '%s:%d: example %d of %d in %s raised %s: %s\n'
                '    >>> %s'
                % (os.path.relpath(test.filename), lineno, i,
                   len(test.examples), test.name,
                   type(exc).__name__, exc,
                   example.source.strip().replace('\n', '\n    ... '))
            ) from exc


@pytest.mark.parametrize('test', DOCTESTS, ids=lambda test: test.name)
def test_docstring_example_runs(test):
    """
    Execute one docstring's examples, discarding their output.

    """
    __tracebackhide__ = True
    printoptions = np.get_printoptions()
    try:
        _run_examples(test)
    finally:
        # Several `game_theory` examples call `np.set_printoptions` for
        # readability and do not restore it; print options are global.
        np.set_printoptions(**printoptions)


def test_failure_report_names_the_example():
    """
    A broken example is reported by file, line, position and source.

    This path only runs when an example is broken, so exercise it with
    a synthetic docstring rather than wait for a real one. The docstring
    below is declared to start at line 10 of `fake.py` (zero-based 9),
    so its sixth line, the broken example, is line 15 of that file. The
    example documenting a traceback must be tolerated on the way there.

    """
    docstring = '''
    >>> x = 1
    >>> raise ValueError('documented')
    Traceback (most recent call last):
    ValueError: documented
    >>> undefined_name + x
    '''
    test = doctest.DocTestParser().get_doctest(
        docstring, globs={}, name='synthetic', filename='fake.py', lineno=9)
    with pytest.raises(AssertionError) as info:
        _run_examples(test)
    message = str(info.value)
    assert message.startswith(
        'fake.py:15: example 3 of 3 in synthetic raised NameError')
    assert '>>> undefined_name + x' in message
    assert isinstance(info.value.__cause__, NameError)
