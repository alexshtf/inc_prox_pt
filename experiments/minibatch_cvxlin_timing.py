import abc
import math
import time
from typing import Callable

import attr

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from minibatch_cvxlin import MiniBatchConvLinOptimizer, HalfSquared, Logistic
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from common import *

@attr.s
class ExperimentDesc(abc.ABC):
    M: int = attr.attrib()
    N: int = attr.attrib()
    outer: Callable = attr.attrib()
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
    opt = MiniBatchConvLinOptimizer(x, desc.outer())
    loss = 0.
    for i, (vecs, ys) in enumerate(DataLoader(dataset, batch_size=desc.batch_size), start=1):
        step_size = 1. / math.sqrt(i)
        ta, tb = desc.linear_transform(vecs, ys)
        try:
            loss += opt.step(step_size, ta, tb).mean().item()
        except RuntimeError as err:
            pass
    proxpt_time = time.perf_counter() - start_time
    proxpt_loss = loss / len(dataset)

    # sovle a problem using SGD
    start_time = time.perf_counter()
    x = torch.zeros_like(x_star, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD(params=(x,), lr=0.0001)
    loss = 0.
    for i, (vecs, ys) in enumerate(DataLoader(dataset, batch_size=desc.batch_size), start=1):
        opt.zero_grad()

        sample_loss = desc.cost(x, vecs, ys).mean()
        sample_loss.backward()
        opt.step()

        loss += sample_loss.item()
    sgd_time = time.perf_counter() - start_time
    sgd_loss = loss / len(dataset)

    return proxpt_time, sgd_time, proxpt_loss, sgd_loss


results = []
for experiment in tqdm(range(30), desc='Experiment'):
    for M in tqdm([1000, 3000, 6000], desc='Dim', leave=False):
        for N in tqdm([5000, 10000, 15000, 20000, 25000, 30000], desc='Sample size', leave=False):
            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=HalfSquared, linear_transform=ls_linear_transform,
                cost=ls_cost,
                batch_size=8
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Least squares', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 8})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=HalfSquared, linear_transform=ls_linear_transform,
                cost=ls_cost,
                batch_size=16
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Least squares', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 16})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=HalfSquared, linear_transform=ls_linear_transform,
                cost=ls_cost,
                batch_size=32
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Least squares', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 32})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=Logistic, linear_transform=logreg_linear_transform,
                cost=logreg_cost,
                batch_size=8
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Logistic regression', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 8})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=Logistic, linear_transform=logreg_linear_transform,
                cost=logreg_cost,
                batch_size=16
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Logistic regression', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 16})

            ppt_time, sgd_time, _, _ = run_experiment(ExperimentDesc(
                M=M, N=N,
                outer=Logistic, linear_transform=logreg_linear_transform,
                cost=logreg_cost,
                batch_size=32
            ))
            results.append({
                'experiment': experiment, 'dim': M, 'num_of_samples': N,
                'type': 'Logistic regression', 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'batch_size': 32})


df = pd.DataFrame.from_records(results)
df.to_csv('minibatch_cvxlin_timing.csv')

df = pd.read_csv('minibatch_cvxlin_timing.csv') \
    .rename(columns={'prox_pt_time': 'Prox-PT', 'sgd_time': 'SGD',
                     'batch_size': 'Batch Size'})

sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
g = sns.FacetGrid(data=df[df['type'] == 'Least squares'], col='dim', row='Batch Size', palette="Set1",
                  sharex=False, sharey=False, margin_titles=True, despine=False,
                  height=3, aspect=1.2)
plt.suptitle('Least squares proximal point vs. SGD running time (seconds)')
g.map_dataframe(sns.regplot, x='SGD', y='Prox-PT', ci=None)
g.add_legend()
plt.show()

g = sns.FacetGrid(data=df[df['type'] == 'Logistic regression'], col='dim', row='Batch Size', palette="Set1",
                  sharex=False, sharey=False, margin_titles=True, despine=False,
                  height=3, aspect=1.2)
plt.suptitle('Logistic regression proximal point vs. SGD running time (seconds)')
g.map_dataframe(sns.regplot, x='SGD', y='Prox-PT', ci=None)
g.add_legend()
plt.show()
