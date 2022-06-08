Pytorch based implementations of proximal operators for useful functions in machine learning. The code is based on a series of blog posts in my blog - https://alexshtf.github.io. See the `examples` folder for usage examples. 

The project consists of several packages for implementing proximal operators for various function types:
* `cvxlin` for compositions of a convex function onto a linear function. Useful for least squares and logistic regression.
* `cvxlinreg` for regularized variants of the above. Useful for Lasso, or L1 / L2 regularized logistic regression.
* `minibatch_cvxlin` for mini-batch variants of a convex onto linear composition. Useful for training least squares of logistic regression models using proximal operators applied to mini-batches of loss functions.
* `cvxlips_quad` for a composition of a convex and Lipschitz function onto a quadratic function. Useful for problems such as phase retrieval, or factorization machines for CTR prediction.


Example - solving a phase retrieval problem using an incremental **proximal point algorithm**:
```python
import torch
from cvxlips_quad import ConvexLipschitzOntoQuadratic, PhaseRetrievalOracles, AbsValue

x = torch.zero(dim)
opt = ConvexLipschitzOntoQuadratic(x)
for t, (a, b) in enumerate(dataset):
    step_size = compute_step_size(t)
    opt.step(step_size, PhaseRetrievalOracles(a, b), AbsValue())
```
