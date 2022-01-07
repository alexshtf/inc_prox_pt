import torch
import torch.linalg


class HalfSquared:
    def solve_dual(self, P, c):
        m = P.shape[1]  # number of columns = batch size

        # construct lhs matrix P* P + m I
        lhs_mat = torch.mm(P.t(), P)
        lhs_mat.diagonal().add_(m)

        # solve positive-definite linear system using Cholesky factorization
        lhs_factor = torch.linalg.cholesky(lhs_mat)
        rhs_col = c.unsqueeze(1)  # make rhs a column vector, so that cholesky_solve works
        return torch.cholesky_solve(rhs_col, lhs_factor)

    def eval(self, lin):
        return 0.5 * (lin ** 2)