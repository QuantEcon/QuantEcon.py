from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
from odu_vfi import searchProblem
from solution_odu_ex1 import res_wage_operator
from compute_fp import compute_fixed_point

# Set up model and compute the function w_bar
sp = searchProblem(pi_grid_size=50, F_a=1, F_b=1)
pi_grid, f, g, F, G = sp.pi_grid, sp.f, sp.g, sp.F, sp.G
phi_init = np.ones(len(sp.pi_grid)) 
w_bar_vals = compute_fixed_point(res_wage_operator, sp, phi_init)
w_bar = lambda x: interp(x, pi_grid, w_bar_vals)


class Agent(object):
    """
    Holds the employment state and beliefs of an individual agent.
    """

    def __init__(self, pi=1e-3):
        self.pi = pi
        self.employed = 1

    def update(self, H):
        "Update self by drawing wage offer from distribution H."
        if self.employed == 0:
            w = H.rvs()
            if w >= w_bar(self.pi):
                self.employed = 1
            else:
                self.pi = 1.0 / (1 + ((1 - self.pi) * g(w)) / (self.pi * f(w)))


num_agents = 5000
separation_rate = 0.025  # Fraction of jobs that end in each period 
separation_num = int(num_agents * separation_rate)
agent_indices = range(num_agents)
agents = [Agent() for i in range(num_agents)]
sim_length = 600
H = G  # Start with distribution G
change_date = 200  # Change to F after this many periods

unempl_rate = []
for i in range(sim_length):
    print "date = ", i
    if i == change_date:
        H = F
    # Randomly select separation_num agents and set employment status to 0
    np.random.shuffle(agent_indices)
    separation_list = agent_indices[:separation_num]
    for agent_index in separation_list:
        agents[agent_index].employed = 0
    # Update agents
    for agent in agents:
        agent.update(H)
    employed = [agent.employed for agent in agents]
    unempl_rate.append(1 - np.mean(employed))

fig, ax = plt.subplots()
ax.plot(unempl_rate, lw=2, alpha=0.8, label='unemployment rate')
ax.axvline(change_date, color="red")
ax.legend()
fig.show()
