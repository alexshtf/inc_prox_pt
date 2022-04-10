import torch


class FMHelper:
    def __init__(self, num_features, embedding_dim):
        self._num_features = num_features
        self._embedding_dim = embedding_dim
        self._pairwise_dim = embedding_dim * num_features
        self._linear_dim = 1 + num_features
        self._eye = None

    def embedding_dim(self):
        return self._embedding_dim

    def num_features(self):
        return self._num_features

    def linear_dim(self):
        return self._linear_dim

    def pairwise_dim(self):
        """The total dimension of the vectors v_1, ..., v_d"""
        return self._pairwise_dim

    def total_dim(self):
        """The total dimension of the model's parameters"""
        return self._pairwise_dim + self._linear_dim

    def eye_embedding_dim(self):
        """An identity matrix of size embedding_dim x embedding_dim"""
        if self._eye is None:
            self._eye = torch.eye(self._embedding_dim, self._embedding_dim)
        return self._eye

    def decompose(self, x):
        """
        Decomposes the parameter vector x of dimension `total_dim` into the bias w_0,
        the linear coefficients w, and the matrix of embedding vectors as its _columns_.
        """
        w_0, w, nu = x.split_with_sizes([1, self._num_features, self._pairwise_dim])
        vs = nu.view(self._embedding_dim, self._num_features)

        return w_0, w, vs

    def eval(self, x, a):
        """
        Computes the value produced by the factorization machine whose model parameter vector is `x`
        and input features `a`
        """
        w_0, w, vs = self.decompose(x)

        lin_term = torch.sum(w * a)
        vs_mult = vs * a
        quad_term = vs_mult.sum(dim=1).square().sum() - vs_mult.square().sum()

        return w_0 + lin_term + 0.5 * quad_term


class FMCTR:
    def __init__(self, a, y, helper):
        self._a = a
        self._y = y
        self._mask = (a != 0)
        self._helper = helper

    def eval(self, x):
        return -self._y * self._helper.eval(x, self._a)  # g is the output of the FM, multiplied by the label

    def scalar(self):
        return 0.  # the constant of g(x) is zero.

    def dual_eval(self, s, step_size, x):
        z_0, z_biases, z_vecs = self.decompose_z_masked(s, step_size, x)
        sol_0, sol_biases, sol_vecs = self.solve_system_masked(s, step_size, x)

        result = -0.5 * (
            z_0 * sol_0 + (z_biases * sol_biases).sum() + (z_vecs * sol_vecs).sum()
        )
        return result.item()

    def dual_deriv(self, s, step_size, x):
        raise NotImplementedError("At this stage, we don't suppose dual derivatives. We don't need them.")

    def solve_system(self, s, step_size, x):
        mask = self._mask
        w_0, w, vs = self._helper.decompose(x)
        w_star_0, w_star_masked, vs_star_masked = self.solve_system_masked(s, step_size, x)

        w_star = w.clone().detach()
        vs_star = vs.clone().detach()

        w_star[mask] = w_star_masked
        vs_star[:, mask] = vs_star_masked

        return torch.cat([w_star_0, w_star, vs_star.view(-1)])

    def solve_system_masked(self, s, step_size, x):
        a = self._a[self._mask]
        y = self._y

        z_0, z_biases, z_vecs = self.decompose_z_masked(s, step_size, x)

        w_star_0 = step_size * z_0
        w_star = step_size * z_biases

        diagonal = step_size / (1 + step_size * y * s * a.square())
        r = (diagonal * a).reshape(-1, 1)
        gamma_sum_part = step_size * y * s * (a.square() / (1 + step_size * y * s * a.square()))
        gamma = (y * s) / (1 - gamma_sum_part.sum())

        times_diag = z_vecs * diagonal  # equivalent to torch.mm(vs, torch.diag(diagonal))
        times_rank_one = z_vecs @ r @ r.T
        vs_star = times_diag + gamma * times_rank_one

        return w_star_0, w_star, vs_star

    def decompose_z_masked(self, s, step_size, x):
        """
        z_0, z_biases, and z_vecs, excluding zero features - used to make the implementation
        fast for sparse input vectors, as discussed in the paper.
        """
        a = self._a[self._mask]
        y = self._y

        w_0, w, vs = self._helper.decompose(x)
        w = w[self._mask]
        z_vecs = vs[:, self._mask]

        z_0 = w_0 / step_size + s * y
        z_biases = w / step_size + s * y * a

        return z_0, z_biases, z_vecs / step_size
