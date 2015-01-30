import pylab
from random import normalvariate, uniform


def generate_data(n, generator_type):
    epsilon_values = []
    for i in range(n):
        if generator_type == "U":
            e = uniform(0, 1)
        else:
            e = normalvariate(0, 1)

        epsilon_values.append(e)
    return epsilon_values

ts_length = 100
data = generate_data(ts_length, 'U')
pylab.plot(data, 'b-')
pylab.show()
