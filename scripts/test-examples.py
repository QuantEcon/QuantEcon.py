#!/usr/bin/python
"""
Test script for QuantEcon executables
=====================================
    examples/*.py 
    solutions/*.ipynb

This script uses a context manager to redirect stdout and stderr
to capture runtime errors for writing to the log file. It also
reports basic execution statistics on the command line (pass/fail)

Usage
-----
python test.py

Default Logs 
------------
    examples/*.py => example-tests.log
"""

import sys
import os
import glob
import subprocess
import re

from common import RedirectStdStreams

set_backend = "import matplotlib\nmatplotlib.use('Agg')\n"

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

def example_tests(test_dir='examples/', log_path='../scripts/example-tests.log'):
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
                sys.stdout.flush()
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
                sys.stdout.flush()
    #-Report-#
    print "[examples/*.py] Passed %i/%i: " %(len(passed), len(test_files))
    if len(failed) == 0:
    	print "Failed Files:\n\tNone"
    else:
    	print "Failed Files:\n\t" + '\n\t'.join(failed)
    print ">> See %s for details" % log_path
    os.chdir('../')
    return passed, failed


if __name__ == '__main__':
    print "-------------------------"
    print "Running all examples/*.py"
    print "-------------------------"
    example_tests(*sys.argv[1:])