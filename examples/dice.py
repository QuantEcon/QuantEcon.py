"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: dice.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""
import random

class Dice:

    faces = (1, 2, 3, 4, 5, 6)

    def __init__(self):
       self.current_face = 1

    def roll(self):
        self.current_face = random.choice(Dice.faces)

