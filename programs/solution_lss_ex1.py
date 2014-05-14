
import numpy as np
import matplotlib.pyplot as plt
from lss import LSS

phi_0, phi_1, phi_2 = 1.1, 0.8, -0.8

A = [[1,     0,     0],
     [phi_0, phi_1, phi_2],
     [0,     1,     0]]
C = np.zeros((3, 1))
G = [0, 1, 0]

ar = LSS(A, C, G, mu_0=np.ones(3))
x, y = ar.simulate(ts_length=50)

fig, ax = plt.subplots(figsize=(8, 4.6))
y = y.flatten()
ax.plot(y, 'b-', lw=2, alpha=0.7)
ax.grid()
ax.set_xlabel('time')
ax.set_ylabel(r'$y_t$', fontsize=16)
plt.show()
