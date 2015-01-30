"""
Filename: quadmap_class.py
Authors: John Stachurski, Thomas J. Sargent

"""


class QuadMap(object):

    def __init__(self, initial_state):
        self.x = initial_state

    def update(self):
        "Apply the quadratic map to update the state."
        self.x = 4 * self.x * (1 - self.x)

    def generate_series(self, n):
        """
        Generate and return a trajectory of length n, starting at the
        current state.
        """
        trajectory = []
        for i in range(n):
            trajectory.append(self.x)
            self.update()
        return trajectory
