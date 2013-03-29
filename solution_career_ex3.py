import matplotlib.pyplot as plt
from matplotlib import cm
from career import *
from compute_fp import compute_fixed_point

wp = workerProblem()
v_init = np.ones((wp.N, wp.N))*100
v = compute_fixed_point(bellman, wp, v_init)
optimal_policy = get_greedy(wp, v)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
tg, eg = np.meshgrid(wp.theta, wp.epsilon)
lvls=(0.5, 1.5, 2.5, 3.5)
ax.contourf(tg, eg, optimal_policy.T, levels=lvls, cmap=cm.winter, alpha=0.5)
ax.contour(tg, eg, optimal_policy.T, colors='k', levels=lvls, linewidths=2)
ax.set_xlabel('theta', fontsize=14)
ax.set_ylabel('epsilon', fontsize=14)
ax.text(1.8, 2.5, 'new life', fontsize=14)
ax.text(4.5, 2.5, 'new job', fontsize=14, rotation='vertical')
ax.text(4.0, 4.5, 'stay put', fontsize=14)
fig.show()
