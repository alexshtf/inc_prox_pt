Pytorch based implementations of proximal operators for useful functions in machine learning. The code is based on a series of blog posts in my blog - https://alexshtf.github.io. See the `examples` folder for usage examples. 

The project consists of several packages for implementing proximal operators for various function types:
* `cvxlin` for compositions of a convex function onto a linear function. Useful for least squares and logistic regression.
* `cvxlinreg` for regularized variants of the above. Useful for Lasso, or L1 / L2 regularized logistic regression.
* `minibatch_cvxlin` for mini-batch variants of a convex onto linear composition. Useful for training least squares of logistic regression models using proximal operators applied to mini-batches of loss functions.
* `cvxlips_quad` for a composition of a convex and Lipschitz function onto a quadratic function. Useful for problems such as phase retrieval, or factorization machines for CTR prediction (experimental)


Example - solving a phase retrieval problem using an incremental **proximal point algorithm**:
```python
import math
import torch
from cvxlinreg import IncRegularizedConvexOnLinear, Logistic, L2Reg

w = torch.zero(dim)
opt = IncRegularizedConvexOnLinear(w, Logistic(), L2Reg(0.01)) # L2 regularized logistic regression
for t, (x, y) in enumerate(dataset):
    step_size = 1 / math.sqrt(t)
    opt.step(step_size, -y * x, 0)  # ln(1 + exp(-y * <w, x> + 0))
```
