import pylab
from random import normalvariate

def generate_data(n):
    epsilon_values = []   
    for i in range(n):
        e = normalvariate(0, 1)
        epsilon_values.append(e)
    return epsilon_values

data = generate_data(100)
pylab.plot(data, 'b-')
pylab.show()
