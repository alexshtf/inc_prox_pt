import torch
from scipy.optimize import minimize_scalar


class ConvexLipschitzOntoQuadratic:
    def __init__(self, x):
        self._x = x

    def step(self, step_size, cvx_oracles, quad_oracles):
        x = self._x
        loss = cvx_oracles.eval(quad_oracles.eval(x))

        def q(s):
            return quad_oracles.dual_eval(s, step_size, x) - cvx_oracles.eval_conj(s) + \
                   s * quad_oracles.scalar()

        def q_prime(s):
            return quad_oracles.dual_deriv(s, step_size, x) + cvx_oracles.conjugate_prime(s) + \
                   quad_oracles.scalar()


        # compute an initial maximization interval
        if cvx_oracles.conjugate_has_compact_domain():
            l, u = cvx_oracles.domain()
        else:
            # scan left until a positive derivative is found
            l = next(s for s in cvx_oracles.lower_bound_sequence() if q_prime(s) > 0)

            # scan right until a negative derivative is found
            u = next(s for s in cvx_oracles.upper_bound_sequence() if q_prime(s) < 0)

        # compute a maximizer of q
        min_result = minimize_scalar(lambda s: -q(s), bounds=(l, u), method='bounded')
        s_star = min_result.x

        # recover the primal optimal solution
        x.set_(quad_oracles.solve_system(s_star, step_size, d))

        return loss