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
        the linear coefficients w, and the matrix of embedding vectors as its rows.
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
        self._helper = helper

    def eval(self, x):
        return -self._y * self._helper.eval(x, self._a)

    def scalar(self):
        return 0.

    def compute_d(self, x, step_size):
        y = self._y
        a = self._a
        helper = self._helper

        z = x / step_size
        z[0] += y
        z[1:helper.linear_dim()] += y * a
        return z

    def mult_mat(self, p):
        helper = self._helper
        a = self._a
        y = self._y

        bias_part, lin_part, pairwise_part = helper.decompose(p)
        amat_tilde = -y * a.outer(a)
        amat_tilde.diagonal().zero_()

        # pairwise_part has the latent vectors in its rows. we need to transpose
        # to embed them into the columns, to conform to the vec() operator.
        pairwise_result = torch.mm(amat_tilde, pairwise_part.t())
        pairwise_result = pairwise_result.t().reshape(-1)  # inverse-vec() operator

        return torch.cat([
            torch.zeros(helper.linear_dim()),
            pairwise_result
        ])

    def solve_system(self, s, eta, p):
        a = self._a
        y = self._y
        helper = self._helper

        bias_part, lin_part, pairwise_part = helper.decompose(p)

        z = eta * y * s * (a ** 2)
        d = eta / (1 - z)
        r = (eta * a) / (1 - z)
        gamma = (y * s) / (1 - torch.sum(z / (1 - z)))

        u = pairwise_part.t()  # vec()
        times_d = u * d
        times_r_rt = torch.mm(torch.mm(u, r.t()), r)
        inverse_product = times_d + gamma * times_r_rt

        return torch.cat([
            eta * bias_part,
            eta * lin_part,
            inverse_product.t().reshape(-1)  # inverse-vec()
        ])
