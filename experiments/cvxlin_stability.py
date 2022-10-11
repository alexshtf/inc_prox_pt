import abc
import math
from functools import partial
from typing import Callable

import attr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.auto import tqdm

from common import *
from cvxlin import IncConvexOnLinear, HalfSquared, Logistic


@attr.s
class ExperimentDesc(abc.ABC):
    type: str = attr.attrib()
    outer: Callable = attr.attrib()
    linear_transform: Callable = attr.attrib()
    cost: Callable = attr.attrib()
    step_size: float = attr.attrib()
    dataset: Dataset = attr.attrib()


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
    opt = IncConvexOnLinear(x, desc.outer())
    loss = 0.
    for i, (a, y) in enumerate(DataLoader(desc.dataset, shuffle=True), start=1):
        step_size = desc.step_size / math.sqrt(i)
        ta, tb = desc.linear_transform(a.squeeze(), y.squeeze())
        loss += opt.step(step_size, ta, tb)
    proxpt_loss = loss / len(ls_dataset)

    x = torch.zeros_like(x_star, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD(params=(x,), lr=desc.step_size)
    sched = LambdaLR(opt, lambda sample: desc.step_size / math.sqrt(1 + sample))
    loss = 0.
    for vecs, ys in DataLoader(desc.dataset, shuffle=True):
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
    prob(step_size=step_size.item())
    for prob in [ls, logreg]
    for step_size in torch.logspace(-2, 1, steps=30)
]


if __name__ == '__main__':
    with mp.Pool(8) as pool:
        results = []
        for repetition in tqdm(range(30), desc='Repetition'):
            tuples = pool.map(run_experiment, experiment_descs)
            for desc, (ppt_loss, sgd_loss) in zip(experiment_descs, tuples):
                results.append({
                    'repetition': repetition, 'step size': desc.step_size,
                    'type': desc.type, 'algorithm': 'proximal point', 'loss': ppt_loss})
                results.append({
                    'repetition': repetition, 'step size': desc.step_size,
                    'type': desc.type, 'algorithm': 'gradient', 'loss': sgd_loss})

    df = pd.DataFrame.from_records(results)
    df.to_csv('cvxlin_stability.csv')

    df = pd.read_csv('cvxlin_stability.csv')
    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='step size', y='loss', row='type',
                    kind="line", hue='algorithm', palette="Set1",
                    facet_kws=dict(sharex=False, sharey=False))
    g.set(xscale="log")
    g.set(yscale="log")
    g.set(ylim=(0.1, 1000))

    plt.show()
