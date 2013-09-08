"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: white_noise_plot.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""

from pylab import plot, show, legend
from random import normalvariate
x = [normalvariate(0, 1) for i in range(100)]
plot(x, 'r-', label="white noise")
legend()
show()
