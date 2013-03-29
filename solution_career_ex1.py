""" 
Simulate job / career paths and compute the waiting time to permanent job /
career.  In reading the code, recall that optimal_policy[i, j] = policy at
(theta_i, epsilon_j) = either 1, 2 or 3; meaning 'stay put', 'new job' and
'new life'.
"""
import matplotlib.pyplot as plt
import numpy as np
from discreterv import discreteRV
from career import *
from compute_fp import compute_fixed_point

wp = workerProblem()
v_init = np.ones((wp.N, wp.N))*100
v = compute_fixed_point(bellman, wp, v_init)
optimal_policy = get_greedy(wp, v)
F = discreteRV(wp.F_probs)
G = discreteRV(wp.G_probs)

def gen_path(T=20):
    i = j = 0  
    theta_index = []
    epsilon_index = []
    for t in range(T):
        if optimal_policy[i, j] == 1:    # Stay put
            pass
        elif optimal_policy[i, j] == 2:  # New job
            j = int(G.draw())
        else:                            # New life
            i, j  = int(F.draw()), int(G.draw())
        theta_index.append(i)
        epsilon_index.append(j)
    return wp.theta[theta_index], wp.epsilon[epsilon_index]

theta_path, epsilon_path = gen_path()
fig = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(epsilon_path, label='epsilon')
ax1.plot(theta_path, label='theta')
ax1.legend(loc='lower right')
theta_path, epsilon_path = gen_path()
ax2 = plt.subplot(212)
ax2.plot(epsilon_path, label='epsilon')
ax2.plot(theta_path, label='theta')
ax2.legend(loc='lower right')
fig.show()
