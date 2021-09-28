from itertools import count


class HalfSquared:
    def eval(self, z):
        return (z ** 2) / 2

    def conjugate_has_compact_domain(self):
        return False

    def lower_bound_sequence(self):
        return (-(2 ** j) for j in count())

    def upper_bound_sequence(self):
        return ((2 ** j) for j in count())

    def conjugate(self, s):
        return (s ** 2) / 2

    def conjugate_prime(self, s):
        return s