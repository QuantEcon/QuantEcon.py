from matplotlib import pyplot as plt
from quantecon import ConsumerProblem

m = ConsumerProblem()
K = 80

# Bellman iteration 
V, c = m.initialize()
print "Starting value function iteration"
for i in range(K):
    print "Current iterate = " + str(i)
    V = m.bellman_operator(V)  
c1 = m.bellman_operator(V, return_policy=True)  

# Policy iteration 
print "Starting policy function iteration"
V, c2 = m.initialize()
for i in range(K):
    print "Current iterate = " + str(i)
    c2 = m.coleman_operator(c2)

fig, ax = plt.subplots()
ax.plot(m.asset_grid, c1[:, 0], label='value function iteration')
ax.plot(m.asset_grid, c2[:, 0], label='policy function iteration')
ax.set_xlabel('asset level')
ax.set_ylabel('consumption (low income)')
ax.legend(loc='upper left')
plt.show()

