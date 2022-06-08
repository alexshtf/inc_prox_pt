import math
from scipy.optimize import root_scalar

class Logistic:
    def solve_dual(self, alpha, beta, tol=1e-16):
        def qprime(s):
            return -alpha * s + beta + math.log1p(-s) - math.log(s)

        # compute [l,u] containing a point with zero qprime
        l = 0.5
        while qprime(l) <= 0:
            l /= 2

        u = 0.5
        while qprime(1 - u) >= 0:
            u /= 2
        u = 1 - u

        sol = root_scalar(qprime, bracket=(l, u), method='brentq')
        return sol.root

    def eval(self, z):
        return math.log(1 + math.exp(z))