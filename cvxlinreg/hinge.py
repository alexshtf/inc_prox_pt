import math


class Hinge:
    def eval(self, z):
        return max(0, z)

    def conjugate_has_compact_domain(self):
        return True

    def domain(self):
        return (0, 1)

    def conjugate(self, s):
        if s < 0 or s > 1:
            return math.inf
        else:
            return 0