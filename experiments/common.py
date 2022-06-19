import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def ls_linear_transform(a, y):
    return a, -y


def ls_cost(x, vecs, ys):
    return (torch.mv(vecs, x) - ys).square() / 2.


def logreg_linear_transform(a, y):
    if a.dim() == 2:
        prod = (a.T * (-y)).T
        return prod, torch.zeros_like(a[0])
    else:
        return -y * a, 0


def logreg_cost(x, vecs, ys):
    return binary_cross_entropy_with_logits(torch.mv(vecs, x), (ys + 1) / 2, reduction='none')
