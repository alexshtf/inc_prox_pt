import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from minibatch_cvxlin import MiniBatchConvLinOptimizer, HalfSquared

# generate a two-dimensional least-squares problem
N = 1000
x_star = torch.tensor([2., -5.])  # the "true" optimal solution
mtx = torch.rand((N, 2)) # A randomly generated data matrix
rhs = torch.mv(mtx, x_star) + torch.distributions.Normal(0, 0.02).sample((N,)) # the RHS, contaminated by noise

# attempt to recover x_star using a L2 (Tikhonov) regularized least squares problem.
x = torch.zeros(x_star.shape)
optimizer = MiniBatchConvLinOptimizer(x, HalfSquared())
dataset = TensorDataset(mtx, rhs)
for epoch in range(5):
    epoch_loss = 0.
    for t, (a, b) in enumerate(DataLoader(dataset, batch_size=10), start=1):
        eta = 1. / math.sqrt(t)
        epoch_loss += optimizer.step(eta, a, b).mean().item()

    print('Epoch loss = ', (epoch_loss / N))

print('Model parameters = ', x)