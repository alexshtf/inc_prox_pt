class Hinge:
    def solve_dual(self, alpha, beta):
        return max(0, min(1, beta / alpha))

    def eval(self, z):
        return max(0, z)