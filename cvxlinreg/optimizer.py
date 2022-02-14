from scipy.optimize import minimize_scalar
import torch


class IncRegularizedConvexOnLinear:
    def __init__(self, x, h, r):
        self._x = x
        self._prox_alg = ProxRegularizedConvexOnLinear(h, r)

    def step(self, eta, a, b):
        x = self._x
        prox_alg = self._prox_alg

        x_new, loss = prox_alg.prox_eval(x, eta, a, b)
        x.set_(x_new)

        return loss


class ProxRegularizedConvexOnLinear:
    def __init__(self, h, r):
        self._h = h
        self._r = r

    def eval(self, x, a, b):
        h = self._h
        r = self._r

        if torch.is_tensor(b):
            b = b.item()

        lin_coef = (torch.dot(a, x) + b).item()
        loss = h.eval(lin_coef) + r.eval(x).item()

        return loss

    def prox(self, x, eta, a, b):
        h = self._h
        r = self._r

        if torch.is_tensor(b):
            b = b.item()

        lin_coef = (torch.dot(a, x) + b).item()
        quad_coef = eta * a.square().sum().item()

        def qprime(s):
            prox = r.prox(eta, x - eta * s * a)
            return torch.dot(a, prox).item() \
                   - h.conjugate_prime(s) \
                   + b

        def q(s):
            return r.envelope(eta, x - eta * s * a) \
                   + lin_coef * s \
                   - quad_coef * (s ** 2) \
                   - h.conjugate(s)

        if h.conjugate_has_compact_domain():
            l, u = h.domain()
        else:
            # scan left until a positive derivative is found
            l = next(s for s in h.lower_bound_sequence() if qprime(s) > 0)

            # scan right until a negative derivative is found
            u = next(s for s in h.upper_bound_sequence() if qprime(s) < 0)

        min_result = minimize_scalar(lambda s: -q(s), bounds=(l, u), method='bounded')
        s_prime = min_result.x
        return r.prox(eta, x - eta * s_prime * a)

    def prox_eval(self, x, eta, a, b):
        return self.prox(x, eta, a, b), self.eval(x, a, b)

    def moreau_envelope(self, x, eta, a, b):
        prox = self.prox(x, eta, a, b)
        return self.eval(prox, a, b) + (x - prox).square().sum() / (2 * eta)
