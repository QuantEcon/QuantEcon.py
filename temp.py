
from ifp import *
import numpy as np
import matplotlib.pyplot as plt

num_iter = 25

fig, ax = plt.subplots()

rs = np.linspace(0, 0.04, 4)
for r in rs:
    cp = consumerProblem(r=r)
    v, c = initialize(cp)
    print r
    for i in range(num_iter):
        c = coleman_operator(cp, c)
    ax.plot(cp.asset_grid, c[:,0], label=str(r))
ax.legend(loc='upper left')
fig.show()
