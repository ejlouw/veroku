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
        assert len(var_names) == 5
        super().__init__(var_names=var_names)
        self.parameter_variable_names = var_names[:4]
        self.data_variable_names = var_names[4:]
        self._dim = len(self.data_variable_names)

    def observe(self, vrs, values):
        if vrs != [self.var_names[-1]]:
            raise NotImplementedError(
                "UnknownGaussian1D.observe only has support for data observations"
            )
        assert len(values) == len(vrs)
        obs_dict = dict(zip(vrs, values))
        x_i = _factor_utils.make_column_vector(obs_dict[self.data_variable_name])
        log_weight_over_norm_const = (-self._dim /2)*math.log(2*math.pi)
        reduced_result = GaussianWishart(
            v=self._dim + 1,
            inv_V=0,
            lambda_0=1,
            mu_0=x_i,
            var_names=self.parameter_variable_names,
            log_weight_over_norm_const=log_weight_over_norm_const)
        return reduced_result



