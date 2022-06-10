from .regularizer import Regularizer
from torch.nn.functional import softshrink
from torch.linalg import vector_norm


class L1Reg(Regularizer):
    def __init__(self, mu):
        self._mu = mu

    def prox(self, eta, x):
        return softshrink(x, eta * self._mu)

    def eval(self, x):
        return self._mu * vector_norm(x, ord=1)
