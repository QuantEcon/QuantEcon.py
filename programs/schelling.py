## Filename: seg.py
## Author: John Stachurski

from random import uniform
from math import sqrt
import matplotlib.pyplot as plt

num_of_type_0 = 200
num_of_type_1 = 200
num_neighbors = 10      # Number of agents regarded as neighbors
require_same_type = 5   # Want at least this many neighbors to be same type


class Agent:

    def __init__(self, type):
        self.type = type
        self.draw_location()

    def draw_location(self):
        self.location = uniform(0, 1), uniform(0, 1)

    def get_distance(self, other):
        "Computes euclidean distance between self and other agent."
        a = (self.location[0] - other.location[0])**2
        b = (self.location[1] - other.location[1])**2
        return sqrt(a + b)

    def happy(self, agents):
        "True if sufficient number of nearest neighbors are of the same type."
        distances = []
        # distances is a list of pairs (d, agent), where d is distance from
        # agent to self
        for agent in agents:
            if self != agent:
                distance = self.get_distance(agent)
                distances.append((distance, agent))
        # Sort from smallest to largest, according to distance
        distances.sort()
        # And extract the neighboring agents
        neighbors = [agent for d, agent in distances[:num_neighbors]]
        # Count how many neighbors have the same type as self
        num_same_type = sum(self.type == agent.type for agent in neighbors)
        return num_same_type >= require_same_type

    def update(self, agents):
        "If not happy, then randomly choose new locations until happy."
        while not self.happy(agents):
            self.draw_location()

def plot_distribution(agents, figname):
    "Plot the distribution of agents in file figname.png."
    x_values_0, y_values_0 = [], []
    x_values_1, y_values_1 = [], []
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_values_0.append(x)
            y_values_0.append(y)
        else:
            x_values_1.append(x)
            y_values_1.append(y)
    plt.plot(x_values_0, y_values_0, 'o', markerfacecolor='orange', markersize=6)
    plt.plot(x_values_1, y_values_1, 'o', markerfacecolor='green', markersize=6)
    plt.savefig(figname)
    plt.clf()


def main():
    """
    Create a list of agents.  Loop until none wishes to move given the 
    current distribution of locations.
    """
    agents = [Agent(0) for i in range(num_of_type_0)]
    agents.extend(Agent(1) for i in range(num_of_type_1))

    count = 1
    while 1:
        print 'Entering loop ', count
        plot_distribution(agents, 'fig%s.png' % count)
        count += 1
        no_one_moved = True
        for agent in agents:
            old_location = agent.location
            agent.update(agents)
            if agent.location != old_location:
                no_one_moved = False
        if no_one_moved:
            break

