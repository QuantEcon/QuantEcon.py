"""
Filename: dice.py
"""

import random


class Dice:

    faces = (1, 2, 3, 4, 5, 6)

    def __init__(self):
        self.current_face = 1

    def roll(self):
        self.current_face = random.choice(Dice.faces)
