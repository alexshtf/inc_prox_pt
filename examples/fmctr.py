import math
from random import Random

import torch
from torch.utils.data import TensorDataset

from cvxlips_quad import ConvexLipschitzOntoQuadratic, Logistic, FMHelper, FMCTR

# factorization machine features
fields = {
    'gender': ['male', 'female', 'unknown_gender'],
    'age': ['0-20', '21-40', '41-inf', 'unknown_age'],
    'device': ['mobile', 'desktop', 'unknown_device'],
    'content': ['finance', 'fashion', 'others']
}
features = [item for lst in fields.values() for item in lst]
num_features = len(features)
feat_idx = dict(zip(features, range(num_features)))
random_gen = Random(737462)


# data generation utilities
def gen_user_item():
    return [
        random_gen.choice(lst)
        for lst in fields.values()
    ]


def make_tensor(user_item):
    a = torch.zeros(num_features)
    indices = torch.tensor([feat_idx[feat] for feat in user_item])
    a[indices] = 1.
    return a


def label_gen(user_item):
    label_rules = [
        (set(), 0.05),
        ({'male'}, 0.05),
        ({'female'}, 0.3),
        ({'0-20'}, 0.05),
        ({'21-40'}, 0.1),
        ({'41-inf'}, 0.4),
        ({'fashion'}, 0.05),
        ({'finance'}, 0.1),
        ({'41-inf', 'fashion'}, 0.2),
    ]

    sum_prob = 0.
    for rule, prob in label_rules:
        if rule.issubset(user_item):
            sum_prob += prob

    if random_gen.uniform(0, 1) <= sum_prob:
        return 1., -math.log(sum_prob)
    else:
        return -1., -math.log(1 - sum_prob)


def data_gen(n):
    aa = []
    ys = []
    ls = []
    for i in range(n):
        user_item = gen_user_item()
        aa.append(make_tensor(user_item))
        label, loss = label_gen(user_item)
        ys.append(label)
        ls.append(loss)

    aa = torch.vstack(aa)
    ys = torch.tensor(ys)
    return TensorDataset(aa, ys), sum(ls) / len(ls)


torch.set_printoptions(linewidth=240, precision=3)
dataset, opt_loss = data_gen(8000)

# model = torch.nn.Linear(num_features, 1)
# criterion = torch.nn.BCEWithLogitsLoss()
# opt = torch.optim.Adam(model.parameters())
# for epoch in range(15):
#     epoch_loss = 0.
#     epoch_label = 0.
#     for a, y in dataset:
#         opt.zero_grad()
#
#         label = (y + 1) / 2.
#         loss = criterion(model.forward(a), label.unsqueeze(-1))
#         epoch_loss += loss.item()
#         epoch_label += label.item()
#
#         loss.backward()
#         opt.step()
#
#     print(f'loss = {epoch_loss / len(dataset)}, opt_loss = {opt_loss}, ctr = {epoch_label / len(dataset)}')
# print(list(model.parameters()))

# loss = 0.5426253185868263, opt_loss = 0.5346452483922727, ctr = 0.3685
# tensor([[-0.515,  0.744, -0.819, -0.645, -0.392,  1.257, -0.919, -0.118, -0.220, -0.135,  0.119,  0.193, -0.455]])
# tensor([-0.046], requires_grad=True)]

helper = FMHelper(num_features, embedding_dim=5)
step_size = 0.05 * 1. / (1 + len(fields))
x = torch.zeros(helper.total_dim())
torch.nn.init.normal_(x, std=1e-2)

w_0, w, vs = helper.decompose(x)
print(w_0)
print(w)
print(vs)

optimizer = ConvexLipschitzOntoQuadratic(x)
outer = Logistic()
for epoch in range(5):
    epoch_label = 0.
    epoch_loss = 0.
    for a, y in dataset:
        if y.item() > 0:
            epoch_label += 1

        inner = FMCTR(a, y, helper)
        epoch_loss += optimizer.step(step_size, outer, inner, debug=False)

    print(f'loss = {epoch_loss / len(dataset)}, opt_loss = {opt_loss}, ctr = {epoch_label / len(dataset)}')

w_0, w, vs = helper.decompose(x)
print(w_0)
print(w)
print(vs)
