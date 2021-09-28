class HalfSquared:
    def solve_dual(self, alpha, beta):
        return beta / (1 + alpha)

    def eval(self, z):
        return 0.5 * (z ** 2)