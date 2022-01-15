import torch


class PhaseRetrievalOracles:
    def __init__(self, a, y):
        self._a = a
        self._y = y

    def eval(self, x):
        return torch.sum(self._a * x) - self._y

    def scalar(self):
        return self._y

    def compute_d(self, x, step_size):
        return x / step_size

    def mult_mat(self, p):
        return torch.sum(self._a * p) * self._a

    def solve_system(self, s, eta, p):
        return eta * (
                p - 2 * eta * s * self.mult_mat(p) / (1 + 2 * eta * s * self._a.square().sum())
        )
