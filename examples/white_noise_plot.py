from pylab import plot, show, legend
from random import normalvariate

x = [normalvariate(0, 1) for i in range(100)]
plot(x, 'b-', label="white noise")
legend()
show()
