from matplotlib import pyplot as plt
from ifp import *

m = consumerProblem()
K = 80

# Bellman iteration 
V, c = initialize(m)
print "Starting value function iteration"
for i in range(K):
    print "Current iterate = " + str(i)
    V = bellman_operator(m, V)  
c1 = bellman_operator(m, V, return_policy=True)  

# Policy iteration 
print "Starting policy function iteration"
V, c2 = initialize(m)
for i in range(K):
    print "Current iterate = " + str(i)
    c2 = coleman_operator(m, c2)

fig, ax = plt.subplots()
ax.plot(m.asset_grid, c1[:, 0], label='value function iteration')
ax.plot(m.asset_grid, c2[:, 0], label='policy function iteration')
ax.set_xlabel('asset level')
ax.set_ylabel('consumption (low income)')
ax.legend(loc='upper left')
plt.show()

