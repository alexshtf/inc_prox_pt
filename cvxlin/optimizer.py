import torch


class IncConvexOnLinear:
    def __init__(self, x, h):
        self._h = h
        self._x = x

    def step(self, eta, a, b):
        """
        Performs the optimizer's step, and returns the loss incurred.
        """
        h = self._h
        x = self._x

        # compute the dual problem's coefficients
        alpha = eta * torch.sum(a ** 2)
        beta = torch.dot(a, x) + b

        # solve the dual problem
        s_star = h.solve_dual(alpha.item(), beta.item())

        # update x
        x.sub_(eta * s_star * a)

        return h.eval(beta.item())

    def x(self):
        return self._x
