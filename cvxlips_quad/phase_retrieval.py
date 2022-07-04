import torch


class PhaseRetrievalOracles:
    def __init__(self, a, y):
        self._a = a
        self._y = y.item() if torch.is_tensor(y) else y

    def eval(self, x):
        return torch.inner(self._a, x).square().item() - self._y

    def scalar(self):
        return -self._y

    def dual_eval(self, s, step_size, x):
        g = x / step_size
        result = -0.5 * torch.inner(g, self.solve_system(s, step_size, x))
        return result.item() - s * self._y

    def dual_deriv(self, s, step_size, x):
        z = self.solve_system(s, step_size, x)
        return torch.inner(z, self._a).square().item() - self._y

    def solve_system(self, s, step_size, x):
        t = 2 * step_size * s
        a = self._a
        numerator = t * (a * x).sum()
        denominator = 1 + t * a.square().sum()
        return x - (numerator / denominator) * a
