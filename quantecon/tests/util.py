"""
Utilities for testing within quantecon

"""
import sys
import os
from contextlib import contextmanager
import numpy as np

if sys.version_info[0] == 2:
    from cStringIO import StringIO
else:  # python 3
    from io import StringIO


@contextmanager
def capture(command, *args, **kwargs):
    """
    A context manager to capture std out, so we can write tests that
    depend on messages that are printed to stdout

    References
    ----------
    http://schinckel.net/2013/04/15/capture-and-test-sys.stdout-sys.
    stderr-in-unittest.testcase/

    Examples
    --------
    class FooTest(unittest.TestCase):
        def test_printed_msg(self):
            with capture(func, *args, **kwargs) as output:
                self.assertRegexpMatches(output, 'should be in print msg')

    """
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


def get_data_dir():
    "Return directory where data is stored"
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "data")
    return data_dir


def max_abs_diff(a1, a2):
    "return max absolute difference between two arrays"
    return np.max(np.abs(a1 - a2))
