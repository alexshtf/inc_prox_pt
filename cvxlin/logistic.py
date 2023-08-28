import math
from scipy.optimize import brentq

class Logistic:
    def solve_dual(self, alpha, beta):
        def qprime(s):
            return -alpha * s + beta + math.log1p(-s) - math.log(s)

        # # compute [l,u] containing a point with zero qprime
        l = 0.5
        while qprime(l) <= 0:
            l /= 4

        u = 0.5
        while qprime(1 - u) >= 0:
            u /= 4
        u = 1 - u

        solution = brentq(qprime, l, u)
        return solution

    def eval(self, z):
        return math.log(1 + math.exp(z))