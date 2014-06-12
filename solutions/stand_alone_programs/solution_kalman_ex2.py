import numpy as np
import matplotlib.pyplot as plt
from quantecon.kalman import Kalman
from scipy.stats import norm
from scipy.integrate import quad

## Parameters
theta = 10
A, G, Q, R = 1, 1, 0, 1
x_hat_0, Sigma_0 = 8, 1
epsilon = 0.1
## Initialize Kalman filter
kalman = Kalman(A, G, Q, R)
kalman.set_state(x_hat_0, Sigma_0)

T = 600
z = np.empty(T)
for t in range(T):
    # Record the current predicted mean and variance, and plot their densities
    m, v = kalman.current_x_hat, kalman.current_Sigma
    m, v = float(m), float(v)
    f = lambda x: norm.pdf(x, loc=m, scale=np.sqrt(v))
    integral, error = quad(f, theta - epsilon, theta + epsilon)
    z[t] = 1 - integral
    # Generate the noisy signal and update the Kalman filter
    kalman.update(theta + norm.rvs(size=1))

fig, ax = plt.subplots()
ax.set_ylim(0, 1)
ax.set_xlim(0, T)
ax.plot(range(T), z) 
ax.fill_between(range(T), np.zeros(T), z, color="blue", alpha=0.2) 
plt.show()

