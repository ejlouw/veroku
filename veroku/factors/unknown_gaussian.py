import math

import numpy as np

from veroku.factors._factor import Factor
from veroku.factors.gaussian_wishart import GaussianWishart
from veroku.factors import _factor_utils


class UnknownGaussian(Factor):
    """
    A Gaussian distribution with unknown mean and variance.
    """

    def __init__(self, var_names):
        """

        :param var_names: The variable names for the alpha, beta, mu_0, kappa_0 and x_i parameters,
            in that order.
        """
        assert len(var_names) > 2
        super().__init__(var_names=var_names)
        self.parameter_variable_names = var_names[:2]
        self.data_variable_names = var_names[2:]
        self._dim = len(self.data_variable_names)

    def observe(self, vrs, values):
        if vrs != [self.var_names[-1]]:
            raise NotImplementedError(
                "UnknownGaussian1D.observe only has support for data observations"
            )
        assert len(values) == len(vrs)
        obs_dict = dict(zip(vrs, values))
        if len(set(vrs) - set(self.data_variable_names)) > 0:
            msg = f"Some provided variable names ({vrs}) not in factor var_names ({self.data_variable_names})"
            raise ValueError(msg)
        if set(vrs) != set(self.data_variable_names):
            msg = "Currently only observation of all data variables is supported"
            raise NotImplementedError(msg)
        x_i = _factor_utils.make_column_vector([obs_dict[v] for v in self.data_variable_names])
        log_weight_over_norm_const = (-self._dim /2)*math.log(2*math.pi)
        reduced_result = GaussianWishart(
            v=self._dim + 1,
            inv_V=0,
            lambda_0=1,
            mu_0=x_i,
            var_names=self.parameter_variable_names,
            log_weight_over_norm_const=log_weight_over_norm_const)
        return reduced_result

    def copy(self):
        factor_copy = UnknownGaussian(var_names=self.var_names)
        return factor_copy



