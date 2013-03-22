from matplotlib import pyplot as plt
from jv import workerProblem 
import numpy as np

# Set up
wp = workerProblem(grid_size=25)

def xbar(phi):
    return (wp.A * phi**wp.alpha)**(1 / (1 - wp.alpha))

phi_grid = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
ax.set_xlabel(r'$\phi$', fontsize=16)
ax.plot(phi_grid, [xbar(phi) * (1 - phi) for phi in phi_grid], 'b-', label=r'$w^*(\phi)$')
ax.legend(loc='upper left')
fig.show()
