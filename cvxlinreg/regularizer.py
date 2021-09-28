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
        result = self.eval(prox) + 0.5 * (prox - x).square().sum() / eta
        return result.item()
