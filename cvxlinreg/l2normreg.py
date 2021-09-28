from .regularizer import Regularizer
from torch.linalg import norm


class L2NormReg(Regularizer):
    def __init__(self, mu):
        self._mu = mu

    def prox(self, eta, x):
        nrm = norm(x)
        eta = eta * self._mu
        return (1 - eta / max(eta, nrm)) * x

    def eval(self, x):
        return self._mu * norm(x)