"""
Utilities for testing within quantecon

@author : Spencer Lyon
@date : 2014-08-01 10:56:32

"""
import sys
from cStringIO import StringIO
from contextlib import contextmanager


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
