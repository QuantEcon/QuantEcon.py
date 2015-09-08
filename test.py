#!/usr/bin/python
"""
Test script for QuantEcon executables
=====================================
    examples/*.py 
    solutions/*.ipynb

This script uses a context manager to redirect stdout and stderr
to capture runtime errors for writing to the log file. It also
reports basic execution statistics (pass/fail)

Usage
-----
python test.py

Default Logs 
------------
    examples/*.py => example-tests.log
    solutions/*.ipynb => solutions-tests.log

"""

import sys
import os
import glob
from contextlib import contextmanager
import traceback
import subprocess
import re

set_backend = "import matplotlib\nmatplotlib.use('Agg')\n"

class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def generate_temp(fl):
    """
    Modify file to supress matplotlib figures
    Preserve __future__ imports at front of file for python intertpreter
    """
    doc = open(fl).read()
    doc = set_backend+doc
    #-Adjust Future Imports-#
    if re.search(r"from __future__ import division", doc):
        doc = doc.replace("from __future__ import division", "")
        doc = "from __future__ import division\n" + doc
    return doc

def example_tests(test_dir='examples/', log_path='example-tests.log'):
    """
    Execute each Python Example File and check exit status.
    The stdout and stderr is also captured and added to the log file
    """
    os.chdir(test_dir)
    test_files = glob.glob('*.py')
    test_files.sort()
    passed = []
    failed = []
    with open(log_path, 'w') as f:
        for i,fname in enumerate(test_files):
            print("Checking program %s (%s/%s) ..."%(fname,i,len(test_files)))
            with RedirectStdStreams(stdout=f, stderr=f):
                print("---Executing '%s'---" % fname)
                #-Generate tmp File-#
                tmpfl = "_" + fname
                fl = open(tmpfl,'w')
                fl.write(generate_temp(fname))
                fl.close()
                #-Run Program-#
                exit_code = subprocess.call(["python",tmpfl], stderr=f)
                if exit_code == 0:
                    passed.append(fname)
                else:
                    failed.append(fname)
                #-Remove tmp file-#
                os.remove(tmpfl)
                print("---END '%s'---" % fname)
    #-Report-#
    print "[examples/*.py] Passed %i/%i: " %(len(passed), len(test_files))
    print "Failed Files:\n\t" + '\n\t'.join(failed)
    print ">> See %s for details" % log_path
    os.chdir('../')
    return passed, failed

def solutions_tests(test_dir, log_path='solutions-tests.log'):
    """
    **IN-WORK**
    """
    test_files = glob.glob(os.path.join(test_dir, '*.ipynb'))
    test_files.sort()
    passed = []
    failed = []
    with open(log_path, 'w') as f:
        with redirected_output(new_stdout=f, new_stderr=f):
            for fname in test_files:
                print("---Executing '%s'---" % fname)
                try:
                    subprocess.call(['runipy',fname], stdout=f, stderr=f)
                except:
                    print("FAIL: %s"%fname)


if __name__ == '__main__':
    example_tests(*sys.argv[1:])