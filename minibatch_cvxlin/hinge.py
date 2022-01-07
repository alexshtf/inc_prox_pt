import torch
import torch.nn.functional
import cvxpy as cp


class Hinge:
    def solve_dual(self, P, c):
        # extract information and convert tensors to numpy. CVXPY
        # works with numpy arrays
        dtype = P.dtype
        m = P.shape[1]
        P = P.data.numpy()
        c = c.data.numpy()

        # define the dual optimization problem using CVXPY
        s = cp.Variable(m)
        objective = 0.5 * cp.sum_squares(P @ s) - cp.sum(cp.multiply(c , s))

        constraints = [s >= 0, s <= 1. / m]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        # solve the problem, and extract the optimal solution
        prob.solve()
        return torch.tensor(s.value).to(dtype=dtype).unsqueeze(1)

    def eval(self, lin):
        return torch.nn.functional.relu(lin)
