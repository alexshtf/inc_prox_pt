import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvxlips_quad import ConvexLipschitzOntoQuadratic, AbsValue, PhaseRetrievalOracles

# generate a two-dimensional least-squares problem
N = 1000
x_star = torch.tensor([2., -5.])  # the label-generating separating hyperplane
features = torch.rand((N, 2))  # A randomly generated data matrix
max_step_size = torch.min(0.5 / features.square().sum(dim=1)).item()

# create binary labels in {-1, 1}
labels = torch.mv(features, x_star).square() + torch.distributions.Laplace(0, 0.02).sample((N,))

# attempt to recover x_star using a L2 (Tikhonov) regularized least squares problem.
x = torch.zeros(x_star.shape)
optimizer = ConvexLipschitzOntoQuadratic(x)
dataset = TensorDataset(features, labels)

outer = AbsValue()
for epoch in range(10):
    epoch_loss = 0.
    for t, (a, b) in enumerate(dataset, start=1):
        eta = max_step_size / math.sqrt(t)
        inner = PhaseRetrievalOracles(a, b)
        epoch_loss += optimizer.step(eta, outer, inner).mean().item()

    print('Epoch loss = ', (epoch_loss / N))

print('Model parameters = ', x)
