import torch
import cvxpy as cp


class Logistic:
    def solve_dual(self, P, c):
        # extract information and convert tensors to numpy. CVXPY
        # works with numpy arrays
        dtype = P.dtype
        m = P.shape[1]
        P = P.data.numpy()
        c = c.data.numpy()

        # define the dual optimization problem using CVXPY
        s = cp.Variable(m)
        objective = 0.5 * cp.sum_squares(P @ s) - \
            cp.sum(cp.multiply(c, s)) - \
            (cp.sum(cp.entr(m * s)) + cp.sum(cp.entr(1 - m * s))) / m

        prob = cp.Problem(cp.Minimize(objective))

        # solve the problem, and extract the optimal solution
        prob.solve()

        # recover optimal solution, and ensure it's cast to the same type as
        # the input data.
        return torch.tensor(s.value).to(dtype=dtype).unsqueeze(1)

    def eval(self, lin):
        return torch.log1p(torch.exp(lin))
