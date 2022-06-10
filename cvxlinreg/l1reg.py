from .regularizer import Regularizer
from torch.nn.functional import softshrink


class L1Reg(Regularizer):
    def __init__(self, mu):
        self._mu = mu

    def prox(self, eta, x):
        return softshrink(x, eta * self._mu)

    def eval(self, x):
        return self._mu * x.abs().sum()
