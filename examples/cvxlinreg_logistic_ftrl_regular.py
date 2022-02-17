import math
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cvxlinreg import ProxRegularizedConvexOnLinear, Logistic, L2Reg


def sparse_random_sample(height, total, occupied):
    return torch.vstack(list(map(
        lambda tensor: tensor.index_select(1, torch.randperm(total)),
        torch.hstack([
            torch.randn(height, occupied),
            torch.zeros(height, total - occupied)
        ]).hsplit(1)
    )))


reg_coef = 0.01
loss_oracle = ProxRegularizedConvexOnLinear(Logistic(), L2Reg(reg_coef))  # L2 Regulaized logistic regression
results = []

for exp in range(10):
    # generate a two-dimensional logistic regression problem
    N = 40
    Ms = [5, 4, 6, 8]
    M = sum(Ms)
    Ms_sparse = [int(0.5 * M) for M in Ms]

    x_star = torch.randn(M) + 0.5  # the label-generating separating hyperplane
    samples = [sparse_random_sample(N, M, int(0.2 * M)) for M in Ms]
    features = torch.hstack(samples)
    print(features.__repr__())

    # create binary labels in {-1, 1}
    labels = torch.mv(features, x_star) + \
             torch.distributions.Normal(0, 0.02).sample((N,))  # the labels, contaminated by noise
    print(f'Positive rate = {torch.heaviside(labels, torch.tensor([1.])).mean().item()}')

    labels = 2 * torch.heaviside(labels, torch.tensor([1.])) - 1
    dataset = TensorDataset(features, labels)

    for eta_zero in torch.logspace(-2, 2, 20):
        # regular FTRL
        grad_sum = torch.zeros_like(x_star)
        x = torch.zeros_like(x_star, requires_grad=True)
        epoch_loss = 0.
        for t, (f, y) in enumerate(dataset, start=1):
            if x.grad is not None:
                x.grad.requires_grad_(False)
                torch.zero_(x.grad)

            loss = torch.log1p(torch.exp(-y * (x * f).sum()))
            epoch_loss += (loss + reg_coef * x.square().sum()).item()

            loss.backward()

            with torch.no_grad():
                grad_sum += x.grad
                eta_curr = math.sqrt(t) / eta_zero
                x.set_(-grad_sum / (eta_curr + reg_coef))
        print(f'Linearized FTRL: eta_zero = {eta_zero}, epoch loss = {(epoch_loss / N)}')
        results.append(dict(algo='FTRL', exp=exp, eta_zero=eta_zero.item(), loss=epoch_loss / N))

        # run "Tight bound" FTRL
        theta = torch.zeros_like(x_star)
        x = torch.zeros_like(x_star)
        epoch_loss = 0.
        for t, (f, y) in enumerate(dataset, start=1):
            a = -y * f  # in logistic regression, the losses are ln(1+exp(-y*f))
            b = torch.zeros(1)

            epoch_loss += loss_oracle.eval(x, a, b)

            lambda_next = math.sqrt(t + 1) / eta_zero
            prox_arg = theta / lambda_next
            prox_scale = 1. / lambda_next
            x, _ = loss_oracle.prox_eval(prox_arg, prox_scale, a, b)
            theta = lambda_next * x
        print(f'Tight bound FTRL: eta_zero = {eta_zero}, epoch loss = {(epoch_loss / N)}')
        results.append(dict(algo='Tight prox FTRL', exp=exp, eta_zero=eta_zero.item(), loss=epoch_loss / N))

        # run "Regular bound" FTRL
        x = torch.zeros_like(x_star)
        epoch_loss = 0.
        for t, (f, y) in enumerate(dataset, start=1):
            a = -y * f  # in logistic regression, the losses are ln(1+exp(-y*f))
            b = torch.zeros(1)

            lambda_curr = math.sqrt(t) / eta_zero
            lambda_next = math.sqrt(t + 1) / eta_zero
            u, loss_val = loss_oracle.prox_eval(x, 1. / lambda_curr, a, b)
            x = u * lambda_curr / lambda_next

            # accumulate loss
            epoch_loss += loss_val
        print(f'Regular bound FTRL: eta_zero = {eta_zero}, epoch loss = {(epoch_loss / N)}')
        results.append(dict(algo='Prox FTRL', exp=exp, eta_zero=eta_zero.item(), loss=epoch_loss / N))

        # run "Regular bound" FTRL + AdaHedge
        x = torch.zeros_like(x_star)
        theta = torch.zeros_like(x_star)
        lambda_curr = 1e-8 / eta_zero
        epoch_loss = 0.
        for t, (f, y) in enumerate(dataset, start=1):
            a = -y * f  # in logistic regression, the losses are ln(1+exp(-y*f))
            b = torch.zeros(1)

            u, loss_val = loss_oracle.prox_eval(x, 1. / lambda_curr, a, b)
            theta_next = u * lambda_curr

            # ada-hedge update
            delta = loss_val - loss_oracle.eval(theta_next / lambda_curr, a, b) - \
                    (theta_next - theta).square().sum() / (2 * lambda_curr)
            lambda_next = lambda_curr + delta / eta_zero

            # update x, theta, and lambda
            x = theta_next / lambda_next
            theta = theta_next
            lambda_curr = lambda_next

            # accumulate loss
            epoch_loss += loss_val
        print(f'AdaHedge FTRL : eta_zero = {eta_zero}, epoch loss = {(epoch_loss / N)}')
        results.append(dict(algo='AdaHedge FTRL', exp=exp, eta_zero=eta_zero.item(), loss=epoch_loss / N))

        # Regular Proximal Point
        x = torch.zeros_like(x_star)
        epoch_loss = 0.
        for t, (f, y) in enumerate(dataset, start=1):
            a = -y * f  # in logistic regression, the losses are ln(1+exp(-y*f))
            b = torch.zeros(1)

            eta_curr = eta_zero / math.sqrt(t)
            x, loss = loss_oracle.prox_eval(x, eta_curr, a, b)
            epoch_loss += loss
        print(f'Proximal point: eta_zero = {eta_zero.item()}, epoch loss = {(epoch_loss / N)}')
        results.append(dict(algo='Proximal Point', exp=exp, eta_zero=eta_zero.item(), loss=epoch_loss / N))

df = pd.DataFrame.from_records(results)
df.to_csv('results.csv', index=False)
df = pd.read_csv('results_2022_02_14_dim100.csv/results.csv', index_col=None)
df = df[df['algo'] != 'FTRL']

plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df, x='eta_zero', y='loss', hue='algo', err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
