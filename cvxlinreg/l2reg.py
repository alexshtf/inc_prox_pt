from .regularizer import Regularizer


class L2Reg(Regularizer):
    def __init__(self, mu):
        self._mu = mu

    def prox(self, eta, x):
        return x / (1 + self._mu * eta)

    def eval(self, x):
        return self._mu * x.square().sum() / 2.
