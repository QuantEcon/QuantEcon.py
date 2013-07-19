from random import uniform

def binomial_rv(n, p):
    count = 0
    for i in range(n):
        U = uniform(0, 1)
        if U < p:
            count = count + 1    # Or count += 1
    print count

