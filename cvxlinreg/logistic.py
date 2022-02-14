import math

class Logistic:
    def eval(self, z):
        if z > 0:
            return math.log1p(math.exp(-z)) + z
        else:
            return math.log1p(math.exp(z))

    def conjugate_has_compact_domain(self):
        return True

    def domain(self):
        return (0, 1)

    def conjugate(self, s):
        def entr(u):
            if u == 0:
                return 0
            else:
                return u * math.log(u)

        return entr(s) + entr(1 - s)