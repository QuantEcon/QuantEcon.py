import pylab
from random import normalvariate

def generate_data(n):
    epsilon_values = []   
    for i in range(n):
        e = normalvariate(0, 1)
        epsilon_values.append(e)
    return epsilon_values

ts_length = 100
data = generate_data(ts_length)
pylab.plot(data, 'b-')
pylab.show()
