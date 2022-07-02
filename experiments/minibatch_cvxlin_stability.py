import abc
import math
from functools import partial
from typing import Callable

import attr

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from minibatch_cvxlin import MiniBatchConvLinOptimizer, HalfSquared, Logistic
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from common import *


@attr.s
class ExperimentDesc(abc.ABC):
    type: str = attr.attrib()
    outer: Callable = attr.attrib()
    linear_transform: Callable = attr.attrib()
    cost: Callable = attr.attrib()
    dataset: Dataset = attr.attrib()
    step_size: float = attr.attrib()
    batch_size: int = attr.attrib(default=32)


M = 100
N = 10000
torch.manual_seed(83483)
x_star = torch.randint(-5, 5, size=(M,), dtype=torch.float32)  # the label-generating separating hyperplane
features = torch.randn((N, M))  # A randomly generated data matrix

# create binary labels in {-1, 1}, contaminated by noise
labels = torch.mv(features, x_star) + torch.distributions.Normal(0, 0.2).sample((N,))
logistic_labels = torch.sign(labels)

# create a dataset
ls_dataset = TensorDataset(features, labels)
logistic_dataset = TensorDataset(features, logistic_labels)


def run_experiment(desc: ExperimentDesc):
    x = torch.zeros_like(x_star, dtype=torch.float32)
    opt = MiniBatchConvLinOptimizer(x, desc.outer())
    loss = 0.
    for i, (vecs, ys) in enumerate(DataLoader(desc.dataset, shuffle=True, batch_size=desc.batch_size), start=1):
        step_size = desc.step_size / math.sqrt(i)
        ta, tb = desc.linear_transform(vecs, ys)
        loss += opt.step(step_size, ta, tb).mean().item()
    proxpt_loss = loss / len(ls_dataset)

    x = torch.zeros_like(x_star, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD(params=(x,), lr=desc.step_size)
    sched = LambdaLR(opt, lambda sample: desc.step_size / math.sqrt(1 + sample))
    loss = 0.
    for vecs, ys in DataLoader(desc.dataset, shuffle=True, batch_size=desc.batch_size):
        x.grad = None

        sample_loss = desc.cost(x, vecs, ys).mean()
        sample_loss.backward()
        opt.step()
        sched.step()

        loss += sample_loss.item()
    sgd_loss = loss / len(logistic_dataset)

    return proxpt_loss, sgd_loss


ls = partial(
    ExperimentDesc,
    type='Least squares',
    outer=HalfSquared,
    linear_transform=ls_linear_transform,
    cost=ls_cost,
    dataset=ls_dataset
)

logreg = partial(
    ExperimentDesc,
    type='Logistic regression',
    outer=Logistic,
    linear_transform=logreg_linear_transform,
    cost=logreg_cost,
    dataset=logistic_dataset
)

experiment_descs = [
    prob(step_size=step_size.item(), batch_size=batch_size)
    for prob in [ls, logreg]
    for step_size in torch.logspace(-2, 2, steps=30)
    for batch_size in [8, 16, 32]
]


if __name__ == '__main__':
    with mp.Pool(8) as pool:
        results = []
        for repetition in tqdm(range(30), desc='Repetition'):
            tuples = pool.map(run_experiment, experiment_descs)
            for desc, (ppt_loss, sgd_loss) in zip(experiment_descs, tuples):
                results.append({
                    'repetition': repetition, 'step size': desc.step_size, 'batch size': desc.batch_size,
                    'type': desc.type, 'algorithm': 'proximal point', 'loss': ppt_loss})
                results.append({
                    'repetition': repetition, 'step size': desc.step_size, 'batch size': desc.batch_size,
                    'type': desc.type, 'algorithm': 'gradient', 'loss': sgd_loss})

    df = pd.DataFrame.from_records(results)
    df.to_csv('minibatch_cvxlin_stability.csv')

    df = pd.read_csv('minibatch_cvxlin_stability.csv')
    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='step size', y='loss', row='type', col='batch size',
                    kind="line", hue='algorithm', palette="Set1",
                    facet_kws=dict(sharex=False, sharey=False))
    g.set(xscale="log")
    g.set(yscale="log")
    g.set(ylim=(0.001, 200))

    plt.show()
