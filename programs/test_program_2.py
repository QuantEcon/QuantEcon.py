import pylab
from random import normalvariate
ts_length = 100
epsilon_values = []   
i = 0
while i < ts_length:
    e = normalvariate(0, 1)
    epsilon_values.append(e)
    i = i + 1
pylab.plot(epsilon_values, 'b-')
pylab.show()
