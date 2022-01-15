import torch
from scipy.optimize import minimize_scalar


class ConvexLipschitzOntoQuadratic:
    def __init__(self, x):
        self._x = x

    def step(self, step_size, cvx_oracles, quad_oracles):
        x = self._x
        loss = cvx_oracles.eval(quad_oracles.eval(x))

        d = quad_oracles.compute_d(x, step_size)

        def q(s):
            mat_times_d = quad_oracles.solve_system(s, step_size, d)
            return -torch.sum(d * mat_times_d).item() / 2 \
                   - cvx_oracles.eval_conj(s) + s * quad_oracles.scalar()

        def q_prime(s):
            mat_sq_times_d = quad_oracles.solve_system(
                s, step_size, quad_oracles.solve_system(s, step_size, d))
            return (-s * torch.sum(d * quad_oracles.mult_mat(d)) * torch.sum(d * mat_sq_times_d)
                    + cvx_oracles.conjugate_prime(s) + quad_oracles.scalar()).item()

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
        x.set(quad_oracles.solve_system(s_star, step_size, d))

        return loss