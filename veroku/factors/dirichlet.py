import copy
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.special import gamma
from veroku.factors._factor import Factor
from veroku.factors._factor_utils import beta_func, di_gamma_func, make_column_vector


class Dirichlet(Factor):

    def __init__(self, alphas, weight=None, var_names=None):

        if var_names is None:
            var_names = [f"v_{i}" for i, _ in enumerate(alphas)]
        super().__init__(var_names=var_names)
        self.alphas = make_column_vector(alphas)
        self.weight = weight
        if weight is None:
            self.weight = 1.0

    def __repr__(self):
        s = f"Dirichlet Factor" \
            f"\nalphas: {np.round(self.alphas.ravel(), 3)}" \
            f"\nweight: {self.weight:.3f}"
        return s

    def potential(self, x):
        if hasattr(x, "__len__"):
            x_ = x.copy()
        else:
            # scalar
            x_ = np.array([[x]])

        if x_.shape[0] == len(self.alphas) - 1:
            x_last = 1 - np.sum(x_, axis=0)
            x_ = np.vstack([x_, x_last])

        assert x_.shape[0] == len(self.alphas)
        invalid_mask = np.all(x_ > 0, axis=0).astype(int)

        assert np.allclose(np.sum(x_, axis=0), 1)

        B_alpha = beta_func(self.alphas)
        px = (1 / B_alpha) * np.prod(np.power(x_, self.alphas-1), axis=0)
        px = self.weight*px*invalid_mask
        return px

    def absorb(self, factor):
        if not isinstance(factor, Dirichlet):
            raise NotImplementedError(f"Absorb only implemented for Dirichlet factor type (received {type(factor)})")
        if set(factor.var_names) != set(self.var_names):
            raise NotImplementedError()
        if self.var_names != factor.var_names:
            other_alphas_sorted = [factor.alpas[factor.var_names.index(sv)] for sv in self.var_names]
        else:
            other_alphas_sorted = factor.alphas
        result_alphas = self.alphas + other_alphas_sorted - 1
        result_weight = beta_func(result_alphas) / (
                    beta_func(self.alphas) * beta_func(factor.alphas))
        result_var_names = copy.deepcopy(self.var_names)
        return Dirichlet(alphas=result_alphas, weight=result_weight, var_names=result_var_names)

    def cancel(self, factor):
        if not isinstance(factor, Dirichlet):
            raise NotImplementedError()
        if set(factor.var_names) != set(self.var_names):
            raise NotImplementedError()
        if self.var_names != factor.var_names:
            other_alphas_sorted = [factor.alpas[factor.var_names.index(sv)] for sv in self.var_names]
        else:
            other_alphas_sorted = factor.alphas
        result_alphas = self.alphas - other_alphas_sorted + 1
        result_weight = beta_func(result_alphas) * beta_func(factor.alphas) / beta_func(self.alphas)
        result_var_names = copy.deepcopy(self.var_names)
        return Dirichlet(alphas=result_alphas, weight=result_weight, var_names=result_var_names)

    def sample(self, num_samples):
        samples = np.random.dirichlet(self.alphas.ravel(), size=num_samples)
        return samples


    def kl_divergence(self, other, normalize_factor=True):
        #https://statproofbook.github.io/P/dir-kl.html
        if not normalize_factor:
            raise NotImplementedError()
        sum_alpha_pi = np.sum(self.alphas)
        term_a = np.log(gamma(np.sum(self.alphas)) / gamma(np.sum(other.alpha_vec)))
        term_b = np.sum([np.log(gamma(alpha_qi)/gamma(alpha_pi)) for alpha_pi, alpha_qi in zip(self.alphas, other.alpha_vec)])
        term_c = np.sum([(alpha_pi - alpha_qi) * (
                    di_gamma_func(alpha_pi) - di_gamma_func(sum_alpha_pi))
                         for alpha_pi, alpha_qi in zip(self.alphas, other.alpha_vec)])
        KL_pq = term_a + term_b + term_c
        return KL_pq

    def plot(self, ax=None):
        if self._dim == 2:
            plot = self.plot_2d(ax=ax)
        elif self._dim == 3:
            plot = self.plot_3d(ax=ax)
        else:
            raise NotImplementedError()
        return plot

    def plot_2d(self, ax=None):
        # 1d simplex in 2d
        n = 100
        if ax is None:
            fig, ax = plt.subplots()
        x_space = np.linspace(0, 1, n).reshape([1, -1])
        p = self.potential(x_space)
        plot = ax.plot(x_space.ravel(), p)
        return plot

    def plot_3d(self, ax=None):
        # 2d simplex in 3d
        n = 100
        plt.figure()
        s_vecs = []
        x_space = np.linspace(0, 1, n)
        y_space = np.linspace(0, 1, n)
        xs = []
        ys = []

        for x in x_space:
            for y in y_space:
                if x + y < 1:
                    xs.append(x)
                    ys.append(y)
                    z = 1 - (x + y)
                    s = np.array([[x, y, z]]).T
                    s_vecs.append(s)
        if ax is None:
            fig, ax = plt.subplots()
        s_vecs_array = np.hstack(s_vecs)
        p = self.potential(s_vecs_array)
        # vxy = np.array([xs,ys])

        r_ = np.array([-1, 1, 0])
        r = r_ / np.linalg.norm(r_)
        R = Rotation.from_rotvec(r * np.radians(45)).as_matrix()
        s_vecs_flat_array = R.T @ s_vecs_array

        # theta= np.radians(225)
        # R = [[np.cos(theta), -np.sin(theta)],
        #     [np.sin(theta), np.cos(theta)]]
        # vxyr = R@vxy
        ax.axis('equal')
        plot = ax.scatter(*s_vecs_flat_array[:2, :], c=p)
        return plot


