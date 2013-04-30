import pylab
from random import normalvariate, uniform

def generate_data(n, generator_type):
    epsilon_values = []   
    for i in range(n):
        e = generator_type(0, 1)
        epsilon_values.append(e)
    return epsilon_values

data = generate_data(100, uniform)
pylab.plot(data, 'b-')
pylab.show()

