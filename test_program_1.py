import pylab
from random import normalvariate
ts_length = 100
epsilon_values = []   # An empty list
for i in range(ts_length):
    e = normalvariate(0, 1)
    epsilon_values.append(e)
pylab.plot(epsilon_values, 'b-')
pylab.show()
