from scipy.optimize import minimize_scalar
import torch


class IncRegularizedConvexOnLinear:
    def __init__(self, x, h, r):
        self._x = x
        self._h = h
        self._r = r

    def step(self, eta, a, b):
        x = self._x
        h = self._h
        r = self._r

        if torch.is_tensor(b):
            b = b.item()

        lin_coef = (torch.dot(a, x) + b).item()
        quad_coef = eta * a.square().sum().item()
        loss = h.eval(lin_coef) + r.eval(x).item()

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
        x.set_(r.prox(eta, x - eta * s_prime * a))
        return loss
