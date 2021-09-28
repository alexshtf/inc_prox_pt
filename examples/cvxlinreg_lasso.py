import math

import torch

from cvxlinreg import IncRegularizedConvexOnLinear, HalfSquared, L1Reg

# generate a two-dimensional least-squares problem
N = 1000
x_star = torch.tensor([2., -5.])  # the "true" optimal solution
mtx = torch.rand((N, 2)) # A randomly generated data matrix
rhs = torch.mv(mtx, x_star) + torch.distributions.Laplace(0, 0.02).sample((N,)) # the RHS, contaminated by noise

# convert the data to separate rows and RHS scalars
my_data_set = [(mtx[i, :], -(rhs[i].item())) for i in range(N)]

# attempt to recover x_star using a L1 regularized least squares problem.
x = torch.zeros(x_star.shape)
optimizer = IncRegularizedConvexOnLinear(x, HalfSquared(), L1Reg(0.01))
epoch_loss = 0.
for t, (a, b) in enumerate(my_data_set, start=1):
    eta = 1. / math.sqrt(t)
    epoch_loss += optimizer.step(eta, a, b)

print('Epoch loss = ', (epoch_loss / N))
print('Model parameters = ', x)