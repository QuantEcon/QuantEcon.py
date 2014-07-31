"""
Simple Test Script which can be used to run a directoy of py files

Just run this file using `python $filename`

"""
from subprocess import call
import glob
files = glob.glob("*.py")
for fl in files:
    print "Testing File: %s" % fl
    call(["python", fl])
    print "------------ END (%s) -----------------" % fl
