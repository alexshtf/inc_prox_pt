import abc
import itertools
import math
import time
from functools import partial
from typing import Callable

import torch
import attr

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from cvxlinreg import IncRegularizedConvexOnLinear, HalfSquared, Logistic, Regularizer, L2Reg, L1Reg
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from common import *


@attr.s
class ExperimentDesc(abc.ABC):
    M: int = attr.attrib()
    N: int = attr.attrib()
    type: str = attr.attrib()
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
    labels = torch.sign(labels)

    dataset = TensorDataset(features, labels)

    # solve a problem using incremental proximal point
    with torch.inference_mode():
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
    for i, (vecs, ys) in enumerate(DataLoader(dataset, batch_size=desc.batch_size), start=1):
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


l2ls = partial(
    ExperimentDesc,
    outer=HalfSquared,
    linear_transform=ls_linear_transform,
    reg=L2Reg(0.0004),
    cost=ls_cost,
    type='L2-LS'
)
l1ls = partial(
    ExperimentDesc,
    outer=HalfSquared,
    linear_transform=ls_linear_transform,
    reg=L1Reg(0.02),
    cost=ls_cost,
    type='L1-LS'
)
l2logreg = partial(
    ExperimentDesc,
    outer=Logistic,
    linear_transform=logreg_linear_transform,
    reg=L2Reg(0.0004),
    cost=logreg_cost,
    type='L2-LogReg'
)
l1logreg = partial(
    ExperimentDesc,
    outer=Logistic,
    linear_transform=logreg_linear_transform,
    reg=L1Reg(0.02),
    cost=logreg_cost,
    type='L1-LogReg'
)
experiment_descs = [
    prob(M=M, N=N, batch_size=batch_size)
    for prob in [l2ls, l1ls, l2logreg, l1logreg]
    for M in [1000, 5000, 10000]
    for N in [500, 1500, 2500]
    for batch_size in [1, 16, 32]
]


if __name__ == '__main__':
    with mp.Pool(8) as pool:
        results = []
        for experiment in tqdm(range(30), desc='Repetition'):
            tuples = pool.map(run_experiment, experiment_descs)
            for desc, (ppt_time, sgd_time, _, _) in zip(experiment_descs, tuples):
                results.append({
                    'experiment': experiment, 'dim': desc.M, 'num_of_samples': desc.N,
                    'type': desc.type, 'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'sgd_batch_size': desc.batch_size})

    df = pd.DataFrame.from_records(results)
    df.to_csv('cvxlinreg_experiment.csv')

    df = pd.read_csv('cvxlinreg_experiment.csv')
    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    sns.lmplot(data=df, x='sgd_time', y='prox_pt_time', hue='type',
               col='dim', row='sgd_batch_size', palette="Set1", ci=None, facet_kws=dict(sharex=False, sharey=False))
    plt.show()
