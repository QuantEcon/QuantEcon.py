"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: career_vf_plot.py
Authors: John Stachurski and Thomas Sargent
LastModified: 11/08/2013

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from career import *
from compute_fp import compute_fixed_point

# === solve for the value function === #
wp = workerProblem()
v_init = np.ones((wp.N, wp.N))*100
v = compute_fixed_point(bellman, wp, v_init)

# === plot value function === #
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
tg, eg = np.meshgrid(wp.theta, wp.epsilon)
ax.plot_surface(tg, eg, v.T, rstride=2, cstride=2, cmap=cm.jet, alpha=0.5,
        linewidth=0.25)
ax.set_zlim(150, 200)
ax.set_xlabel('theta', fontsize=14)
ax.set_ylabel('epsilon', fontsize=14)
plt.show()
