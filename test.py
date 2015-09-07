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
python test.py examples/

Default Logs 
------------
    examples/*.py => logs/run-examples.log
    solutions/*.ipynb => logs/run-solutions.log

"""

import sys
import os
import glob
from contextlib import contextmanager
import traceback
import subprocess

set_backend = "import matplotlib\nmatplotlib.use('Agg')\n"

@contextmanager
def redirected_output(new_stdout=None, new_stderr=None):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if new_stdout is not None:
        sys.stdout = new_stdout
    if new_stderr is not None:
        sys.stderr = new_stderr
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr

def example_tests(test_dir='examples/', log_path='example-tests.log'):
    os.chdir(test_dir)
    test_files = glob.glob('*.py')
    test_files.sort()
    passed = []
    failed = []
    with open(log_path, 'w') as f:
        for fname in test_files:
            print("---Executing '%s'---" % fname)
            try:
                sed = subprocess.Popen(["sed", r"2i import matplotlib\nmatplotlib.use('Agg')\n", fname], stdout=subprocess.PIPE)
                exit_code = subprocess.call(["python"], stdin=sed.stdout, stdout=f, stderr=f)
                if exit_code == 1:
                    passed.append(fname)
                else:
                    failed.append(fname)
            print("---END '%s'---" % fname)

    print "[examples/*.py] Passed %i/%i: " %(len(passed), len(test_files))
    print "Failed Files:\n" + '\t\n'.join(failed)
    print
    print ">> See %s for details" % log_path
    os.chdir('../')
    return passed, failed

def solutions_tests(test_dir, log_path='solutions-tests.log'):
    test_files = glob.glob(os.path.join(test_dir, '*.ipynb'))
    test_files.sort()
    passed = []
    failed = []
    with open(log_path, 'w') as f:
        with redirected_output(new_stdout=f, new_stderr=f):
            for fname in test_files:
                print("---Executing '%s'---" % fname)
                try:
                    subprocess.call(['runipy',fname])
                except:
                    print("FAIL: %s"%fname)


if __name__ == '__main__':
    example_tests(*sys.argv[1:])