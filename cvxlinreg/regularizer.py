from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def prox(self, eta, x):
        pass

    @abstractmethod
    def eval(self, x):
        pass

    def envelope(self, eta, x):
        prox = self.prox(eta, x)
        result = self.eval(prox) + (eta / 2) * (prox - x).square().sum()
        return result.item()
