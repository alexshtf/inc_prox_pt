import math


class AbsValue:
    def eval(self, z):
        return abs(z)

    def conjugate_has_compact_domain(self):
        return True

    def domain(self):
        return (-1, 1)

    def conjugate(self, s):
        if -1 <= s <= 1:
            return 0
        else:
            return math.inf
