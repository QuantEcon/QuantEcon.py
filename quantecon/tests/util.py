"""
Utilities for testing within quantecon

"""
import sys
import os
from os.path import join, exists
from contextlib import contextmanager
import numpy as np
import tables

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
    # this_dir = os.path.abspath(".")
    data_dir = os.path.join(this_dir, "data")
    return data_dir


def get_h5_data_file():
    """
    return the data file used for holding test data. If the data
    directory or file do not exist, they are created.

    Notes
    -----
    This should ideally be called from a context manage as so::

        with get_h5_data_file() as f:
            # do stuff

    This way we know the file will be closed and cleaned up properly

    """
    data_dir = get_data_dir()

    if not exists(data_dir):
        os.mkdir(data_dir)

    data_file = join(data_dir, "testing_data.h5")

    return tables.open_file(data_file, "a", "Data for quantecon tests")


def get_h5_data_group(grp_name, parent="/", f=get_h5_data_file()):
    """
    Try to fetch the group named grp_name from the file f. If it doesn't
    yet exist, it is created

    Parameters
    ----------
    grp_name : str
        A string specifying the name of the new group. This should be
        only the group name, not including any information about the
        group's parent (path)

    parent : str, optional(default="/")
        The parent or path for where the group should live. If nothing
        is given, the group will be created at the root node `"/"`

    f : hdf5 file, optional(default=get_h5_data_file())
        The file where this should happen. The default is the data file
        for these tests

    Returns
    -------
    existed : bool
        A boolean specifying whether the group existed or was created

    group : tables.Group
        The requested group

    Examples
    --------
    with get_h5_data_file() as f:
        my_group = get_h5_data_group("jv")  # data for jv tests

    Notes
    -----
    As with other code dealing with I/O from files, it is best to call
    this function within a context manager as shown in the example.

    """
    existed = True
    try:
        group = f.getNode(parent + grp_name)
    except:
        # doesn't exist
        existed = False
        msg = "data for {} tests".format(grp_name + ".py")
        group = f.create_group(parent, grp_name, msg)

    return existed, group


def write_array(f, grp, array, name):
    "stores array in into group grp of h5 file f under name name"
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(grp, name, atom, array.shape)
    ds[:] = array


def max_abs_diff(a1, a2):
    "return max absolute difference between two arrays"
    return np.max(np.abs(a1 - a2))
