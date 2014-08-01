"""
Utilities for testing within quantecon

@author : Spencer Lyon
@date : 2014-08-01 10:56:32

"""
import sys
import os
from os.path import join
from cStringIO import StringIO
from contextlib import contextmanager
import numpy as np
import tables


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
    # this_dir = os.path.abspath(".")
    data_dir = os.path.join(this_dir, "data")
    return data_dir


def get_h5_data_file():
    """
    return the data file used for holding test data

    Notes
    -----
    This should ideally be called from a context manage as so::

        with get_h5_data_file() as f:
            # do stuff

    This way we know the file will be closed and cleaned up properly

    """
    data_dir = get_data_dir()
    data_file = join(data_dir, "testing_data.h5")
    return tables.open_file(data_file, "a", "Data for quantecon tests")


def write_array(f, grp, array, name):
    "stores array in into group grp of h5 file f under name name"
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(grp, name, atom, array.shape)
    ds[:] = array


def max_abs_diff(a1, a2):
    "return max absolute difference between two arrays"
    return np.max(np.abs(a1 - a2))
