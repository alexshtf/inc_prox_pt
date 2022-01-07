import torch
import math


class MiniBatchConvLinOptimizer:
    def __init__(self, x, phi):
        self._x = x
        self._phi = phi

    def step(self, step_size, A_batch, b_batch):
        # helper variables
        x = self._x
        phi = self._phi

        # compute dual problem coefficients
        P = math.sqrt(step_size) * A_batch.t()
        c = torch.addmv(b_batch, A_batch, x)

        # solve dual problem
        s_star = phi.solve_dual(P, c)

        # perform step
        step_dir = torch.mm(A_batch.t(), s_star)
        x.sub_(step_size * step_dir.reshape(x.shape))

        # return the mini-batch losses w.r.t the params before making the step
        return phi.eval(c)
