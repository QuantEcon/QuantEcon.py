from random import uniform

payoff = 0
count = 0

for i in range(10):
    U = uniform(0, 1)
    count = count + 1 if U < 0.5 else 0
    if count == 3:
        payoff = 1

print payoff
