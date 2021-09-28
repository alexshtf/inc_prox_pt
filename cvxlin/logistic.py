import math

class Logistic:
    def solve_dual(self, alpha, beta, tol=1e-16):
        def qprime(s):
            return -alpha * s + beta + math.log(1 - s) - math.log(s)

        # compute [l,u] containing a point with zero qprime
        l = 0.5
        while qprime(l) <= 0:
            l /= 2

        u = 0.5
        while qprime(1 - u) >= 0:
            u /= 2
        u = 1 - u

        # bisection starting from [l, u] above
        while u - l > tol:
            mid = (u + l) / 2
            if qprime(mid) == 0:
                return mid
            if qprime(l) * qprime(mid) > 0:
                l = mid
            else:
                u = mid

        return (u + l) / 2

    def eval(self, z):
        return math.log(1 + math.exp(z))