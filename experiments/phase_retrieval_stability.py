import abc
import time

import attr
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cvxlips_quad import ConvexLipschitzOntoQuadratic, AbsValue, PhaseRetrievalOracles
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


@attr.s
class ExperimentDesc(abc.ABC):
    M: int = attr.attrib()
    N: int = attr.attrib()
    dataset: torch.utils.data.Dataset = attr.attrib()
    step_size: float = attr.attrib()
    max_step_size: float = attr.attrib()
    batch_size: int = attr.attrib(default=1)


def run_experiment(desc: ExperimentDesc):
    dataset = desc.dataset
    N = len(dataset)

    # solve a problem using incremental proximal point
    if desc.step_size <= desc.max_step_size:
        with torch.inference_mode():
            x = torch.zeros((desc.M,), dtype=torch.float32)
            torch.nn.init.normal_(x, std=1e-3)

            opt = ConvexLipschitzOntoQuadratic(x)
            loss = 0.
            abs_value = AbsValue()
            for i, (a, y) in enumerate(desc.dataset, start=1):
                step_size = desc.step_size / math.sqrt(i)
                loss += opt.step(step_size, abs_value, PhaseRetrievalOracles(a, y))
            proxpt_loss = loss / N
    else:
        proxpt_loss = float('NaN')

    # sovle a problem using proximal-SGD
    x = torch.zeros((desc.M,), dtype=torch.float32, requires_grad=True)
    torch.nn.init.normal_(x, std=1e-3)

    opt = torch.optim.SGD(params=(x,), lr=desc.step_size)
    sched = LambdaLR(opt, lambda sample: desc.step_size / math.sqrt(1 + sample))
    loss = 0.
    for i, (vecs, ys) in enumerate(DataLoader(dataset, batch_size=desc.batch_size), start=1):
        opt.zero_grad()

        sample_loss = torch.abs(torch.mv(vecs, x).square() - ys).mean()
        sample_loss.backward()

        opt.step()
        sched.step()

        loss += sample_loss.item() * vecs.shape[0]
    sgd_loss = loss / N

    return proxpt_loss, sgd_loss


def get_dataset(M, N):
    torch.manual_seed(83483)
    x_star = torch.randn(size=(M,), dtype=torch.float32)  # the label-generating separating hyperplane
    features = torch.rand((N, M)) / M  # A randomly generated data matrix

    labels = torch.mv(features, x_star).square() + torch.distributions.Laplace(0, 0.02).sample((N,))
    dataset = TensorDataset(features, labels)
    max_step_size = torch.min(0.5 / features.square().sum(dim=1)).item()

    return dataset, max_step_size


M = 100
N = 10000
ds, max_step_size = get_dataset(M, N)
batch_size = 1
experiment_descs = [
    ExperimentDesc(M=M, N=N, batch_size=batch_size, step_size=step_size, max_step_size=max_step_size, dataset=ds)
    for step_size in list(np.geomspace(1e-1, 100., 32)) + [max_step_size]
]

if __name__ == '__main__':
    with mp.Pool(16) as pool:
        results = []
        for experiment in tqdm(range(30), desc='Repetition'):
            tuples = pool.map(run_experiment, experiment_descs)
            for desc, (ppt_loss, sgd_loss) in zip(experiment_descs, tuples):
                results.append({
                    'experiment': experiment, 'algorithm': 'proximal point', 'step size': desc.step_size, 'loss': ppt_loss})
                results.append({
                    'experiment': experiment, 'algorithm': 'SGD', 'step size': desc.step_size, 'loss': sgd_loss})

    df = pd.DataFrame.from_records(results)
    df.to_csv('phase_retrieval_stability.csv')

    df = pd.read_csv('phase_retrieval_stability.csv')
    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='step size', y='loss', hue='algorithm', kind="line")
    g.set(xscale="log")
    g.set(yscale="log")
    g.set(ylim=(2e-2, 3e-2))
    plt.show()
