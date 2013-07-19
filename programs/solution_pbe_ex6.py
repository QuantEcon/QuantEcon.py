from pylab import plot, show, legend
from random import normalvariate

alphas = [0.0, 0.8, 0.98]
ts_length = 200

for alpha in alphas:
    x_values = []
    current_x = 0
    for i in range(ts_length):
        x_values.append(current_x)
        current_x = alpha * current_x + normalvariate(0, 1)
    plot(x_values, label='alpha = ' + str(alpha))
legend()
show()


