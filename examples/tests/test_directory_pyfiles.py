"""
Simple Test Script which can be used to run a directoy of py files

Just run this file using `python $filename`

"""
#-Subprocess Recipe-#
from subprocess import call
import glob
files = glob.glob("*.py")
for fl in files:
    print "Testing File: %s" % fl
    call(["python", fl])
    print "------------ END (%s) -----------------" % fl

#-IPYTHON NOTEBOOK Recipe-#
#-Instructions-#
#--------------#
#-1. Open an IPython Notebook in quantecon.py/examples/ folder
#-2. Copy the following code recipe into the notebook and run
import glob
files = glob.glob("*.py")
%pylab inline
for fl in files:
    print "----RUNNING (%s)----"%fl
    %run $fl
    print "----END (%s)-----"%fl