"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Filename: quadmap.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""

import pylab

def qm(x, n):
    i = 0
    while i < n:
        yield x
        x = 4 * (1 - x) * x
        i += 1

h = qm(0.1, 200)

time_series = [x for x in h]
pylab.plot(time_series)
pylab.show()


