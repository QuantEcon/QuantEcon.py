'''
	Simple Test Script which can be used to run a directoy of py files
	Usage:
	-----
		cd /dir/files.py
		ipython notebook --pylab=inline
		copy below into first cell to run ALL programs
'''


import glob
files = glob.glob("*.py")
for fl in files:
    print "Testing File: %s" % fl
    %run $fl
    print "------------ END (%s) -----------------" % fl