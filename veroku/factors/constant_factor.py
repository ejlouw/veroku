"""
A module for instantiating and performing operations on multivariate Gaussian distributions.
"""

# pylint: disable=cyclic-import
# pylint: disable=protected-access
# pylint: disable=no-self-use

# System imports
import copy

# Third-party imports
import numpy as np

from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL
# Local imports
from veroku.factors._factor import Factor


class ConstantFactor(Factor):
    """
    A class for instantiating and performing operations on Vacuous Factors (non-normalisable constants).
    """


    def __init__(self, log_constant_value):
        """
        """
        super().__init__(var_names=[])
        self.log_constant_value = log_constant_value

    def equals(self, factor, rtol=DEFAULT_FACTOR_RTOL, atol=DEFAULT_FACTOR_ATOL):
        """
        Check if this factor is equal to another.

        :param factor: The other factor.
        :param rtol: The (numpy) relative tolerance.
        :param atol: The (numpy) absolute tolerance.
        :return: The result of the comparison.
        """
        if np.isclose(self.log_constant_value, factor.log_constant_value, rtol=rtol, atol=atol):
            return True
        return False

    def absorb(self, factor):
        """
        Multiply this factor with another factor.

        :param Gaussian factor: the factor to absorb with
        :return: the resulting factor
        """
        factor_copy = copy.deepcopy(factor)
        factor_copy.log_weight += self.log_constant_value
        return factor_copy

    def cancel(self, factor):
        """
        Divide this factor by another factor.

        :param factor: the factor to cancel by
        :return: the resulting factor
        """
        factor_copy = copy.deepcopy(factor)
        factor_copy.log_weight -= self.log_constant_value
        return factor_copy

    def reduce(self, vrs, values):
        """
        Observe a subset of the variables in the scope of this factor and return the resulting factor.

        :param vrs: the names of the observed variable (list)
        :type vrs: str list
        :param values: the values of the observed variables
        :type values: vector-like
        :return: the resulting Gaussian
        :rtype: Gaussian
        """
        raise NotImplementedError()

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        return 0.0


    def kl_divergence(self, other):
        """
        Get the KL-divergence D_KL(self || factor) between a normalized version of this factor and another factor.

        :param other: The other factor
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        if isinstance(other, ConstantFactor):
            return 0.0
        raise NotImplementedError()

    @property
    def is_vacuous(self):
        """
        Check if a factor distribution contains no information.

        :return: The result of the check.
        :rtype: Bool
        """
        return True

    def copy(self):
        """
        Make a copy of this factor.

        :return: The copied factor.
        :rtype: Gaussian
        """
        return ConstantFactor(log_constant_value=self.log_constant_value)


    def potential(self, x_val):
        """
        Get the value of the Gaussian potential at x_val.

        :param x_val: The vector (or vector-like object) to evaluate the Gaussian at
        :type x_val: numpy.ndarray
        :return: The value of the Gaussian potential at x_val.
        """
        return np.exp(self.log_constant_value)

    def log_potential(self, x_val, vrs=None):
        """
        Get the log of the value of the Gaussian potential at x_val.

        :param x_val: the vector (or vector-like object) to evaluate the Gaussian at
        :type x_val: vector-like
        :param vrs: The variables corresponding to the values in x_val.
        :type vrs: str list
        :return: The log of the value of the Gaussian potential at x_val.
        """
        return self.log_constant_value


    def __repr__(self):
        """
        Get the string representation of the Gaussian factor.

        :return: The factor representation string.
        :rtype: str
        """
        repr_str = (
                "Vacuous Factor \n"
                + f"constant_value = {np.exp(self.log_constant)}"
        )
        return repr_str
