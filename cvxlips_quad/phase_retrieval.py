import torch


class PhaseRetrievalOracles:
    def __init__(self, a, y):
        self._a = a
        self._y = y

    def eval(self, x):
        return torch.sum(self._a * x) - self._y

    def scalar(self):
        return self._y

    def dual_eval(self, s, step_size, x):
        p = x / step_size
        return -0.2 * (p * self.solve_system(s, step_size, p)).sum().item()

    def dual_deriv(self, s, step_size, x):
        g = x / step_size
        z = self.solve_system(s, step_size, g)
        return (z * self._a).sum().square().item()

    def mult_mat(self, p):
        return torch.sum(self._a * p) * self._a

    def solve_system(self, s, step_size, p):
        return step_size * (
                p - 2 * step_size * s * self.mult_mat(p) / (1 + 2 * step_size * s * self._a.square().sum())
        )
