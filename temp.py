from ifp import *
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
cp = consumerProblem()
v, c = initialize(cp)
for i in range(50):
    c = coleman_operator(cp, c)
ax.plot(cp.asset_grid, c[:,0])
fig.show()
