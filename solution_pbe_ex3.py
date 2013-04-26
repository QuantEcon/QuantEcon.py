from random import uniform
from math import sqrt

n = 100000

count = 0
for i in range(n):
    u, v = uniform(0, 1), uniform(0, 1)
    d = sqrt((u - 0.5)**2 + (v - 0.5)**2)
    if d < 0.5:
        count += 1

area_estimate = count / float(n)

print area_estimate * 4  # dividing by radius**2

