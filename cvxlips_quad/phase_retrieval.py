import torch


class PhaseRetrievalOracles:
    def __init__(self, a, y):
        self._a = a
        self._y = y.item() if torch.is_tensor(y) else y

    def eval(self, x):
        return torch.dot(self._a, x).square().item() - self._y

    def scalar(self):
        return -self._y

    def dual_eval(self, s, step_size, x):
        quad_part = torch.dot(x, self.solve_system(s, step_size, x)) / (2 * step_size)
        return -quad_part.item() - s * self._y

    def dual_deriv(self, s, step_size, x):
        z = self.solve_system(s, step_size, x)
        return torch.dot(z, self._a).square().item() - self._y

    def solve_system(self, s, step_size, x):
        t = 2 * step_size * s
        a = self._a
        numerator = t * torch.dot(a, x)
        denominator = 1. + t * torch.dot(a, a)
        return x - (numerator / denominator) * a
