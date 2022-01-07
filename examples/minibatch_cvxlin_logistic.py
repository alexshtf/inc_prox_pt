import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from minibatch_cvxlin import MiniBatchConvLinOptimizer, Logistic

# generate a two-dimensional least-squares problem
N = 10000
x_star = torch.tensor([2., -5.])  # the label-generating separating hyperplane
features = torch.rand((N, 2))  # A randomly generated data matrix

# create binary labels in {-1, 1}
labels = torch.mv(features, x_star) + torch.distributions.Normal(0, 0.02).sample((N,))  # the labels, contaminated by noise
labels = 2 * torch.heaviside(labels, torch.tensor([1.])) - 1

# attempt to recover x_star using a L2 (Tikhonov) regularized least squares problem.
x = torch.zeros(x_star.shape)
optimizer = MiniBatchConvLinOptimizer(x, Logistic())
dataset = TensorDataset(features, labels)
for epoch in range(10):
    epoch_loss = 0.
    for t, (f, y) in enumerate(DataLoader(dataset, batch_size=32), start=1):
        eta = 1. / math.sqrt(t)
        a = -y.unsqueeze(1) * f  # in logistic regression, the losses are ln(1+exp(-y*f))
        b = torch.zeros_like(y)
        epoch_loss += optimizer.step(eta, a, b).mean().item()

    print('Epoch loss = ', (epoch_loss / N))

print('Model parameters = ', x)
