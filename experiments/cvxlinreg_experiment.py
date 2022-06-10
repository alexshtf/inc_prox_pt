import abc
import math
import time
from typing import Callable

import torch
import attr

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cvxlinreg import IncRegularizedConvexOnLinear, HalfSquared, Logistic, Regularizer, L2Reg, L1Reg
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm


@attr.s
class ExperimentDesc(abc.ABC):
    M: int = attr.attrib()
    N: int = attr.attrib()
    outer: Callable = attr.attrib()
    reg: Regularizer = attr.attrib()
    linear_transform: Callable = attr.attrib()
    cost: Callable = attr.attrib()
    batch_size: int = attr.attrib(default=32)


def run_experiment(desc: ExperimentDesc):
    x_star = torch.randint(-5, 5, size=(desc.M,), dtype=torch.float32)  # the label-generating separating hyperplane
    features = torch.rand((desc.N, desc.M))  # A randomly generated data matrix

    # create binary labels in {-1, 1}, contaminated by noise
    labels = torch.mv(features, x_star) + torch.distributions.Normal(0, 0.02).sample((desc.N,))
    labels = 2 * torch.heaviside(labels, torch.tensor([1.])) - 1

    dataset = TensorDataset(features, labels)

    # solve a problem using incremental proximal point
    start_time = time.perf_counter()
    x = torch.zeros_like(x_star, dtype=torch.float32)
    opt = IncRegularizedConvexOnLinear(x, desc.outer(), desc.reg)
    loss = 0.
    for i, (a, y) in enumerate(dataset, start=1):
        step_size = 1. / math.sqrt(i)
        ta, tb = desc.linear_transform(a, y)
        loss += opt.step(step_size, ta, tb)
    proxpt_time = time.perf_counter() - start_time
    proxpt_loss = loss / len(dataset)

    # sovle a problem using proximal-SGD
    start_time = time.perf_counter()
    x = torch.zeros_like(x_star, dtype=torch.float32, requires_grad=True)
    step_size = 0.0001
    opt = torch.optim.SGD(params=(x,), lr=step_size)
    loss = 0.
    for i, (vecs, ys) in enumerate(DataLoader(dataset, batch_size=32), start=1):
        opt.zero_grad()

        sample_loss = desc.cost(x, vecs, ys).mean()
        sample_loss.backward()

        opt.step()

        with torch.no_grad():
            x.set_(desc.reg.prox(step_size, x))

        loss += sample_loss.item()
    sgd_time = time.perf_counter() - start_time
    sgd_loss = loss / len(dataset)

    return proxpt_time, sgd_time, proxpt_loss, sgd_loss


results = []
for experiment in tqdm(range(30), desc='Experiment'):
    for M in tqdm([100, 200, 300, 400, 500, 600], desc='Dim', leave=False):
        for N in tqdm([500, 1000, 1500, 2000, 2500, 3000], desc='Sample size', leave=False):
            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=HalfSquared, linear_transform=lambda a, y: (a, -y),
                reg=L2Reg(0.0004),
                cost=lambda x, vecs, ys: (torch.mv(vecs, x) - ys).square()
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'L2-LS', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=HalfSquared, linear_transform=lambda a, y: (a, -y),
                reg=L1Reg(0.02),
                cost=lambda x, vecs, ys: (torch.mv(vecs, x) - ys).square()
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'L1-LS', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                reg=L2Reg(0.0004),
                outer=Logistic, linear_transform=lambda a, y: (-y * a, torch.zeros(1)),
                cost=lambda x, vecs, ys: torch.binary_cross_entropy_with_logits(torch.mv(vecs, x), (ys + 1) / 2)
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'L2-LogReg', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                reg=L1Reg(0.02),
                outer=Logistic, linear_transform=lambda a, y: (-y * a, torch.zeros(1)),
                cost=lambda x, vecs, ys: torch.binary_cross_entropy_with_logits(torch.mv(vecs, x), (ys + 1) / 2)
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'L2-LogReg', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time})


df = pd.DataFrame.from_records(results)
df.to_csv('cvxlinreg_experiment.csv')

plt.figure(figsize=(16, 12))
df = pd.read_csv('cvxlinreg_experiment.csv')
df = df.drop(df[(df['dim'] == 300) & (df['sgd_time'] > 0.5)].index)
sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
sns.lmplot(data=df, x='sgd_time', y='prox_pt_time', hue='type',
           col='dim', col_wrap=3, palette="Set1", ci=None, facet_kws=dict(sharex=False, sharey=False))
plt.show()
