import abc
import time

import attr
import math
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
    step_size: float = attr.attrib()
    batch_size: int = attr.attrib(default=32)


def run_experiment(desc: ExperimentDesc):
    x_star = torch.randn(size=(desc.M,), dtype=torch.float32)  # the label-generating separating hyperplane
    features = torch.rand((desc.N, desc.M))  # A randomly generated data matrix
    max_step_size = torch.min(0.5 / features.square().sum(dim=1)).item()

    # create binary labels in {-1, 1}, contaminated by noise
    labels = torch.mv(features, x_star).square() + torch.distributions.Laplace(0, 0.02).sample((desc.N,))
    dataset = TensorDataset(features, labels)

    # solve a problem using incremental proximal point
    if desc.step_size > max_step_size:
        proxpt_time = float('nan')
        proxpt_loss = float('nan')
    else:
        with torch.inference_mode():
            start_time = time.perf_counter()
            x = torch.zeros_like(x_star, dtype=torch.float32)
            opt = ConvexLipschitzOntoQuadratic(x)
            loss = 0.
            abs_value = AbsValue()
            for i, (a, y) in enumerate(dataset, start=1):
                step_size = desc.step_size / math.sqrt(i)
                loss += opt.step(step_size, abs_value, PhaseRetrievalOracles(a, y))
            proxpt_time = time.perf_counter() - start_time
            proxpt_loss = loss / len(dataset)

    # sovle a problem using proximal-SGD
    start_time = time.perf_counter()
    x = torch.zeros_like(x_star, dtype=torch.float32, requires_grad=True)
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
    sgd_time = time.perf_counter() - start_time
    sgd_loss = loss / len(dataset)

    return proxpt_time, sgd_time, proxpt_loss, sgd_loss


experiment_descs = [
    ExperimentDesc(M=M, N=N, batch_size=batch_size, step_size=(0.1 / M))
    for M in [1000, 3000, 6000]
    for N in [500, 1000, 1000, 2000, 2500, 3000]
    for batch_size in [1, 16, 32]
]


if __name__ == '__main__':
    with mp.Pool(16) as pool:
        results = []
        for experiment in tqdm(range(30), desc='Repetition'):
            tuples = pool.map(run_experiment, experiment_descs)
            for desc, (ppt_time, sgd_time, _, _) in zip(experiment_descs, tuples):
                results.append({
                    'experiment': experiment, 'dim': desc.M, 'num_of_samples': desc.N,
                    'prox_pt_time': ppt_time, 'sgd_time': sgd_time, 'sgd_batch_size': desc.batch_size})

    df = pd.DataFrame.from_records(results)
    df.to_csv('phase_retrieval_timing.csv')

    df = pd.read_csv('phase_retrieval_timing.csv')
    sns.set(context='paper', palette='Set1', style='ticks', font_scale=1.5)
    plt.figure(figsize=(16, 12))
    sns.lmplot(data=df, x='sgd_time', y='prox_pt_time',
               col='dim', row='sgd_batch_size', palette="Set1", ci=None, facet_kws=dict(sharex=False, sharey=False))
    plt.show()
