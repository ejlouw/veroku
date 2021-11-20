"""
This module contains truncated factor classes and helper functions.
"""

import matplotlib.pyplot as plt

import numpy as np

from veroku.factors._factor import Factor
import copy
from scipy.integrate import quad
from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL
from veroku.factors.experimental.gaussian_mixture import GaussianMixture

def get_supports_intersection(support_bounds_a, support_bounds_b):
    result_support_start = max(support_bounds_a[0], support_bounds_b[0])
    result_support_end = min(support_bounds_a[1], support_bounds_b[1])
    if result_support_start > result_support_end:
        raise NonOverLappingSupportsError()
    result_support_bounds = (result_support_start, result_support_end)
    return result_support_bounds


class NonOverLappingSupportsError(Exception):
    pass

class IntegrationError(Exception):
    pass

#TODO: correct docstrings


class Truncated1DFactor(Factor):

    def __init__(self, non_truncated_factor, support_bounds, normalize):
        """
        The initializer.
        """
        assert support_bounds[0] < support_bounds[1]
        self._support_bounds = copy.deepcopy(support_bounds)
        self.non_truncated_factor = non_truncated_factor.copy()
        self.weight = 1.0
        if normalize:
            self.normalize(inplace=True)

        if len(non_truncated_factor.var_names) > 1:
            raise ValueError("Truncated1DFactor only supports 1 dimensional distributions")
        super().__init__(var_names=non_truncated_factor.var_names)

    @property
    def var_names(self):
        """
        Get the factor's variable names.

        :return: the var_names parameter.
        """
        return copy.deepcopy(self._var_names)

    @property
    def support_bounds(self):
        """
        Get the factor's variable names.

        :return: the var_names parameter.
        """
        return copy.deepcopy(self._support_bounds)

    @property
    def dim(self):
        """
        Get the factor's dimensionality.

        :return: the dim parameter.
        """
        return self._dim

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between the message factor and a vacuous (uniform) version of it.

        :return: The KL-divergence
        """

    def equals(self, factor, rtol=DEFAULT_FACTOR_RTOL, atol=DEFAULT_FACTOR_ATOL):
        """
         An abstract function for checking if this factor is equal to another factor.

        :param factor: The factor to be compared to.
        :param float rtol: The relative tolerance to use for factor equality check.
        :param float atol: The absolute tolerance to use for factor equality check.
        :return: The result of the check.
        :rtype: bool
        """

    def copy(self):
        """
        An abstract function for copying a factor that should be implemented in the base class.

        :return: a copy of the factor
        """
        return Truncated1DFactor(non_truncated_factor=self.non_truncated_factor,
                                 support_bounds=self.support_bounds,
                                 normalize=False)

    def marginalize(self, vrs, keep=True):
        """
        An abstract function for performing factor marginalisation that should be implemented in the base class.

        :return: the resulting marginal factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor marginal.
        """
        assert vrs == self.var_names
        if keep:
            return self.copy()
        return self.get_integral()

    def absorb(self, factor):
        """
        (Alias for multiply) A function for performing factor multiplication.

        :param factor: The factor to be multiplied with.
        :return: The resulting product
        """
        if isinstance(factor, Truncated1DFactor):
            result_support_bounds = get_supports_intersection(self.support_bounds, factor.support_bounds)
            non_truncated_factor_product = self.non_truncated_factor.absorb(factor.non_truncated_factor)
        else:
            result_support_bounds = self.support_bounds
            non_truncated_factor_product = self.non_truncated_factor.absorb(factor)
        return Truncated1DFactor(non_truncated_factor=non_truncated_factor_product,
                                 support_bounds=result_support_bounds,
                                 normalize=False)

    def multiply(self, factor):
        return self.absorb(factor)

    def divide(self, factor):
        """
        An abstract function for performing factor division that should be implemented in the base class.

        :param factor: The factor to be divided by.
        :return: The resulting quotient
        """
        raise NotImplementedError()

    def cancel(self, factor):
        """
        (Alias for divide by default - but can differ in certain cases) An abstract function for performing factor
        division (or division-like operations - see Categorical cancel for example) that can be implemented in the
        base class.

        :param factor: The factor to be divided by.
        :return: The resulting factor
        """
        if isinstance(factor, Truncated1DFactor):
            result_support_bounds = get_supports_intersection(self.support_bounds, factor.support_bounds)
            non_truncated_factor_product = self.non_truncated_factor.cancel(factor.non_truncated_factor)
        else:
            result_support_bounds = self.support_bounds
            non_truncated_factor_product = self.non_truncated_factor.cancel(factor)
        return Truncated1DFactor(non_truncated_factor=non_truncated_factor_product,
                                 support_bounds=result_support_bounds,
                                 normalize=False)

    def is_within_supports(self, x):
        return self.support_bounds[0] < x < self.support_bounds[1]

    def reduce(self, vrs, values):
        """
        An abstract function for performing the observation (a.k.a conditioning) operation that should be implemented
        in the base class.

        :return: the resulting reduced factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param values: The values of vars.
        :return: The resulting reduced factor.
        """
        assert vrs == self.var_names
        return self.potential(values)

    def potential(self, x_val):
        """
        Get the value of the Gaussian potential at x_val.

        :param x_val: The vector (or vector-like object) to evaluate the Gaussian at
        :type x_val: numpy.ndarray
        :return: The value of the Gaussian potential at x_val.
        """
        if isinstance(x_val, np.ndarray):
            assert x_val.shape == (1) or x_val.shape == (1,1)
            x = x_val.ravel()[0]
        else:
            x = x_val  # assume int or float type
        if self.is_within_supports(x):
            return self.weight*self.non_truncated_factor.potential(x)
        return 0.0

    def get_integral(self):
        unweighted_integral_over_bounds, abs_error = quad(self.non_truncated_factor.potential,
                                                      a=self.support_bounds[0],
                                                      b=self.support_bounds[1])
        integration_percentage_error = abs_error/unweighted_integral_over_bounds
        if integration_percentage_error > 0.01:
            raise IntegrationError(f"Large integration error ({integration_percentage_error})")
        integral_over_bounds = self.weight * unweighted_integral_over_bounds
        return integral_over_bounds

    def normalize(self, inplace=False):
        """
        An abstract function for performing factor normalization that should be implemented in the base class.

        :return: The normalized factor.
        """
        if not inplace:
            return Truncated1DFactor(non_truncated_factor=self.non_truncated_factor.copy(),
                                     support_bounds=self.support_bounds,
                                     normalize=True)
        self.weight = 1.0 / self.get_integral()

    def show(self):
        """
        An abstract function for printing the parameters of the factor that should be implemented in the base class.
        """
        support_span = self.support_bounds[1] - self.support_bounds[0]

        xs_supported = np.linspace(self.support_bounds[0],
                                   self.support_bounds[1])

        xs_zero_0 = np.linspace(self.support_bounds[0] - 0.2 * support_span,
                                self.support_bounds[0])
        xs_zero_1 = np.linspace(self.support_bounds[1],
                                self.support_bounds[1] + 0.2 * support_span)
        xs = np.concatenate((xs_zero_0, xs_supported, xs_zero_1))
        p_zero_0 = xs_zero_0 * 0
        p_zero_1 = xs_zero_1 * 0
        p_supported = np.array([self.weight*self.non_truncated_factor.potential(x) for x in xs_supported])
        p = np.concatenate((p_zero_0, p_supported, p_zero_1))
        plt.plot(xs, p)

    def __add__(self, other):
        # TODO: replace Gaussian mixture with general mixture class here
        return GaussianMixture(self.copy(), other.copy())
# TODO: add KLD and distance from vacuous methods
