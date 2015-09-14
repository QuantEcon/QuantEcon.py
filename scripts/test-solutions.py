#!/usr/bin/python
"""
Test solutions/*.ipynb

Notes
-----
  1. This script should be run from the root level "python scripts/test-solutions.py"

"""

import sys
import os
import glob
import subprocess

from common import RedirectStdStreams

def solutions_tests(test_dir='solutions/', log_path='../scripts/solutions-tests.log'):
    """
    Execute each Jupyter Notebook
    """
    os.chdir(test_dir)
    test_files = glob.glob(os.path.join('*.ipynb'))
    test_files.sort()
    passed = []
    failed = []
    with open(log_path, 'w') as f:
            for i,fname in enumerate(test_files):
                print("Checking notebook %s (%s/%s) ..."%(fname,i,len(test_files)))
                with RedirectStdStreams(stdout=f, stderr=f):
                    print("---> Executing '%s' <---" % fname)
                    sys.stdout.flush()
                    #-Run Program-#
                    exit_code = subprocess.call(["runipy",fname], stdout=open(os.devnull, 'wb'), stderr=f)
                    sys.stderr.flush()
                    if exit_code == 0:
                        passed.append(fname)
                    else:
                        failed.append(fname)
                    print("---> END '%s' <---" % fname)
                    print
                    sys.stdout.flush()
    #-Report-#
    print "[solutions/*.py] Passed %i/%i: " %(len(passed), len(test_files))
    if len(failed) == 0:
    	print "Failed Notebooks:\n\tNone"
    else:
    	print "Failed Notebooks:\n\t" + '\n\t'.join(failed)
    print ">> See %s for details" % log_path
    os.chdir('../')
    return passed, failed  


if __name__ == '__main__':
    print "-----------------------------"
    print "Running all solutions/*.ipynb"
    print "-----------------------------"
    solutions_tests(*sys.argv[1:])