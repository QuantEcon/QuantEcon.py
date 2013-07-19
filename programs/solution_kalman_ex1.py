import numpy as np
import matplotlib.pyplot as plt
from kalman import Kalman
from scipy.stats import norm

## Parameters
theta = 10
A, G, Q, R = 1, 1, 0, 1
x_hat_0, Sigma_0 = 8, 1
## Initialize Kalman filter
kalman = Kalman(A, G, Q, R)
kalman.set_state(x_hat_0, Sigma_0)

N = 5
fig, ax = plt.subplots()
xgrid = np.linspace(theta - 5, theta + 2, 200)
for i in range(N):
    # Record the current predicted mean and variance, and plot their densities
    m, v = kalman.current_x_hat, kalman.current_Sigma
    m, v = float(m), float(v)
    ax.plot(xgrid, norm.pdf(xgrid, loc=m, scale=np.sqrt(v)), label=r'$t=%d$' % i)
    # Generate the noisy signal
    y = theta + norm.rvs(size=1)
    # Update the Kalman filter
    kalman.update(y)

ax.set_title(r'First %d densities when $\theta = %.1f$' % (N, theta)) 
ax.legend(loc='upper left')
fig.show()

