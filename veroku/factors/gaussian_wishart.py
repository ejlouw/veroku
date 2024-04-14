import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, multigammaln
from scipy.integrate import dblquad

from veroku.factors import _factor_utils
from veroku.factors._factor import Factor


import numpy as np
from scipy.special import multigammaln


def calculate_gaussian_wishart_normalising_constant(v, inv_V, lambda_0, mu_0):
    # TODO: update to return log

    # 1D check
    #norm_check_1d = 2 ** (v / 2) * np.linalg.det(V)**(v/2)*gamma(v/2)*((2*np.pi)/lambda_0)**0.5

    # real
    d = mu_0.shape[0]
    two_pow_vd_2 = 2 ** (v * d / 2)
    det_V_pow_v_2 = np.linalg.det(inv_V) ** (-v / 2)
    multi_gamma_vd1__2 = np.exp(multigammaln((v - d + 1) / 2, d))
    pi2_pow_d_2 = (2 * np.pi) ** (d / 2)
    numerator = two_pow_vd_2 * det_V_pow_v_2 * multi_gamma_vd1__2 * pi2_pow_d_2
    Z = numerator / (lambda_0**0.5)
    return Z

class GaussianWishart(Factor):
    def __init__(self, v, inv_V, lambda_0, mu_0, var_names, log_weight=None, log_weight_over_norm_const=None):
        super().__init__(var_names=var_names)
        self.v = v
        self.inv_V = _factor_utils.make_square_matrix(inv_V)
        self.lambda_0 = lambda_0
        self.mu_0 = _factor_utils.make_column_vector(mu_0)
        self._dim = self.mu_0.shape[0]

        # TODO: improve this. Maybe only use log_weight_over_norm_const and calculate log weight
        #  when necessary.
        if log_weight_over_norm_const is not None:
            assert log_weight is None
            self.log_weight_over_norm_const = log_weight_over_norm_const
            self.log_weight = None
        elif log_weight is not None:
            assert log_weight_over_norm_const is None
            self.log_weight = log_weight
            weight_over_norm_const = self._calculate_normalising_constant()
            self.log_weight_over_norm_const = np.log(weight_over_norm_const)
        else:
            raise ValueError()

    def get_normal_gamma_param_dict(self):
        normal_gamma_param_dict = {
            "mu_0": self.mu_0[0, 0],
            "kappa_0": self.lambda_0,
            "alpha": self.v / 2,
            "beta": (1 *self.inv_V[0, 0]) / 2,
            "log_weight": self.log_weight,
        }
        return normal_gamma_param_dict

    def __repr__(self):
        s = f"v = {self.v} \ninv_V = {self.inv_V} \nlambda_0 = {self.lambda_0}\nmu_0 = {self.mu_0} \n"
        s = s + f"log_weight = {self.log_weight}"
        return s


    def get_norm_numeric(self):
        result, error = dblquad(self.potential, 1e-10, 5, lambda x2: -20, lambda x2: 20)
        return result

    def kl_divergence(self, other):
           raise NotImplementedError()

    def absorb(self, other):
        assert self._dim == other._dim
        assert self.var_names == other.var_names
        var_names = copy.deepcopy(self.var_names)
        d = self._dim
        inv_V_a = self.inv_V
        inv_V_b = other.inv_V
        mu_a = self.mu_0
        mu_b = other.mu_0
        v_a = self.v
        v_b = other.v
        lambda_a = self.lambda_0
        lambda_b = other.lambda_0

        # Calculate other variables based on the given formulas
        v_c = v_a + v_b - d
        lambda_c = lambda_a + lambda_b
        mu_c = (lambda_a * mu_a +  lambda_b * mu_b)/(lambda_a + lambda_b)

        U = mu_a @ mu_a.T + mu_b @ mu_b.T - 2 * mu_b @ mu_a.T
        lambda_ab_c = (lambda_a * lambda_b) / lambda_c
        inv_V_a_V_b_sum = inv_V_a + inv_V_b
        lambda_ab_c_U = lambda_ab_c * U
        inv_V_c = inv_V_a_V_b_sum + lambda_ab_c_U

        Z_a = self._calculate_normalising_constant()
        Z_b = other._calculate_normalising_constant()
        Z_c = calculate_gaussian_wishart_normalising_constant(
            v=v_c, inv_V=inv_V_c, lambda_0=lambda_c, mu_0=mu_c
        )
        log_weight = self.log_weight + other.log_weight + np.log(Z_c) - (np.log(Z_a) + np.log(Z_b))
        product = GaussianWishart(
            v=v_c, inv_V=inv_V_c, lambda_0=lambda_c, mu_0=mu_c, log_weight=log_weight, var_names=var_names
        )
        return product


    def _calculate_normalising_constant(self):
        Z = calculate_gaussian_wishart_normalising_constant(
            v=self.v, inv_V=self.inv_V, lambda_0=self.lambda_0, mu_0=self.mu_0
        )
        return Z


    # TODO: make log_potential
    def potential(self, mu, K):
        mu = _factor_utils.make_column_vector(mu)
        K = _factor_utils.make_square_matrix(K)
        # TODO: rewrite to make use of _calculate_normalising_constant and check
        # Convenience variables
        two_pow_vd_2 = 2**(self.v * self._dim / 2)
        det_K = np.linalg.det(K)
        v_min_d = self.v - self._dim
        inv_V = np.linalg.inv(self.V)
        det_V = np.linalg.det(self.V)
        v = self.v
        half_v = v/2
        multi_gamma_vd1__2 = np.exp(multigammaln((v_min_d + 1)/2, self._dim))

        # Calculate the first term
        first_term_numerator = det_K**((v_min_d - 1) / 2)
        first_term_denominator = two_pow_vd_2 * (det_V**(half_v)) * multi_gamma_vd1__2
        first_term = first_term_numerator / first_term_denominator

        # Calculate the second term
        exponent2 = -0.5 * np.trace(inv_V @ K)
        second_term = np.exp(exponent2)

        # Calculate the third term
        third_term_numerator = (self.lambda_0 * det_K) ** 0.5
        third_term_denominator = (2*np.pi)**(self._dim/2)
        third_term = third_term_numerator / third_term_denominator

        # Calculate the third term
        lambda0_K = self.lambda_0 * K
        exponent4 = -0.5 * (mu - self.mu_0).T @ lambda0_K @ (mu - self.mu_0)
        fourth_term = np.exp(exponent4)

        # Put it all together
        potential = first_term * second_term * third_term * fourth_term
        return potential
