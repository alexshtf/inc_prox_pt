import abc
import math
from functools import partial
from typing import Callable

import attr
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm

from cvxlinreg import Regularizer, IncRegularizedConvexOnLinear, HalfSquared, L2Reg, Logistic, L1Reg
from common import ls_linear_transform, ls_cost, logreg_linear_transform, logreg_cost


@attr.s
class ExperimentDesc(abc.ABC):
    type: str = attr.attrib()
    outer: Callable = attr.attrib()
    linear_transform: Callable = attr.attrib()
    cost: Callable = attr.attrib()
    step_size: float = attr.attrib()
    reg: Regularizer = attr.attrib()
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
    opt = IncRegularizedConvexOnLinear(x, desc.outer(), desc.reg)
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
    for i, (vecs, ys) in enumerate(DataLoader(desc.dataset, shuffle=True), start=1):
        x.grad = None

        sample_loss = desc.cost(x, vecs, ys).mean()
        reg_value = desc.reg.eval(x)

        sample_loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            step_size = desc.step_size / math.sqrt(i)
            x.set_(desc.reg.prox(step_size, x))

        sample_loss = (sample_loss + reg_value).item()
        loss += sample_loss
    sgd_loss = loss / len(logistic_dataset)

    return proxpt_loss, sgd_loss


l2ls = partial(
    ExperimentDesc,
    type='Least squares - L2Reg',
    outer=HalfSquared,
    reg=L2Reg(0.04),
    linear_transform=ls_linear_transform,
    cost=ls_cost,
    dataset=ls_dataset
)

l1ls = partial(
    ExperimentDesc,
    type='Least squares - L1Reg',
    outer=HalfSquared,
    reg=L1Reg(0.2),
    linear_transform=ls_linear_transform,
    cost=ls_cost,
    dataset=ls_dataset
)

l2logreg = partial(
    ExperimentDesc,
    type='Logistic regression - L2Reg',
    outer=Logistic,
    reg=L2Reg(0.04),
    linear_transform=logreg_linear_transform,
    cost=logreg_cost,
    dataset=logistic_dataset
)

l1logreg = partial(
    ExperimentDesc,
    type='Logistic regression - L1Reg',
    outer=Logistic,
    reg=L1Reg(0.2),
    linear_transform=logreg_linear_transform,
    cost=logreg_cost,
    dataset=logistic_dataset
)

experiment_descs = [
    prob(step_size=step_size.item())
    for prob in [l2ls, l1ls, l2logreg, l1logreg]
    for step_size in torch.logspace(-2, 1, steps=5)
]


if __name__ == '__main__':
    # with mp.Pool(8) as pool:
    #     results = []
    #     for repetition in tqdm(range(30), desc='Repetition'):
    #         tuples = pool.map(run_experiment, experiment_descs)
    #         for desc, (ppt_loss, sgd_loss) in zip(experiment_descs, tuples):
    #             results.append({
    #                 'repetition': repetition, 'step size': desc.step_size,
    #                 'type': desc.type, 'algorithm': 'proximal point', 'loss': ppt_loss})
    #             results.append({
    #                 'repetition': repetition, 'step size': desc.step_size,
    #                 'type': desc.type, 'algorithm': 'gradient', 'loss': sgd_loss})
    #
    # df = pd.DataFrame.from_records(results)
    # df.to_csv('cvxlinreg_stability.csv')
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 10000
    display.max_colwidth = 199
    display.width = 1000
    df = pd.read_csv('cvxlinreg_stability.csv')
    print(df)

    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='step size', y='loss', row='type',
                    kind="line", hue='algorithm', palette="Set1",
                    facet_kws=dict(sharex=False, sharey=False))
    g.set(xscale="log")
    g.set(yscale="log")
    g.set(ylim=(0.1, 1000))

    plt.show()


