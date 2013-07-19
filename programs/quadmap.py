## Filename: quadmap.py
## Author: John Stachurski

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


