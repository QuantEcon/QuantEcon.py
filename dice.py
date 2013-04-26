import random

class Dice:

    faces = (1, 2, 3, 4, 5, 6)

    def __init__(self, current_face):
       self.current_face = current_face

    def roll(self):
        self.current_face = random.choice(Dice.faces)

