"""
This module contains an abstract parent class, defining the minimum functions that all factors should have
"""

from abc import ABCMeta, abstractmethod
import copy

from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL


class Factor:
    """
    An abstract parent class.
    """

    __metaclass__ = ABCMeta

    def __init__(self, var_names):
        """
        A super class constructor that should be called from the base class constructor.
        """
        if len(set(var_names)) != len(var_names):
            raise ValueError("duplicate variables in var_names: ", var_names)

        self._var_names = var_names
        if not isinstance(var_names, list):
            self._var_names = [var_names]

        self._dim = len(var_names)

    @property
    def var_names(self):
        """
        Get the factor's variable names.

        :return: the var_names parameter.
        """
        return copy.deepcopy(self._var_names)

    @property
    def dim(self):
        """
        Get the factor's dimensionality.

        :return: the dim parameter.
        """
        return self._dim

    def get_marginal_vars(self, vrs, keep=True):
        """
        A helper function (for marginalize) that returns the variables that should be marginalised out (keep=False)
        or kept (keep=True).

        :param vrs: (list) the variables
        :param keep: Whether these variables are to be kept or summed out.
        :return: The variables to be kept or summed out.
        """
        if isinstance(vrs, str):
            vrs = [vrs]
        if keep:
            return vrs.copy()
        vrs = [v for v in self.var_names if v not in vrs]
        return vrs

    @abstractmethod
    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between the message factor and a vacuous (uniform) version of it.

        :return: The KL-divergence
        """

    @property
    def is_vacuous(self):
        """
        Check if this factor is vacuous (i.e uniform).

        :return: Whether the factor is vacuous or not.
        :rtype: bool
        """
        if self.distance_from_vacuous() < 1e-10:
            return True
        return False

    @abstractmethod
    def equals(self, factor, rtol=DEFAULT_FACTOR_RTOL, atol=DEFAULT_FACTOR_ATOL):
        """
         An abstract function for checking if this factor is equal to another factor.

        :param factor: The factor to be compared to.
        :param float rtol: The relative tolerance to use for factor equality check.
        :param float atol: The absolute tolerance to use for factor equality check.
        :return: The result of the check.
        :rtype: bool
        """

    @abstractmethod
    def copy(self):
        """
        An abstract function for copying a factor that should be implemented in the base class.

        :return: a copy of the factor
        """

    @abstractmethod
    def marginalize(self, vrs, keep=True):
        """
        An abstract function for performing factor marginalisation that should be implemented in the base class.

        :return: the resulting marginal factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor marginal.
        """

    @abstractmethod
    def absorb(self, factor):
        """
        An abstract function for performing factor multiplication (with the same type of factor)
        that should be implemented in the base class.

        :param factor: The factor to be multiplied with.
        :return: The resulting product
        """

    @abstractmethod
    def cancel(self, factor):
        """
        An abstract function for performing factor division (by the same type of factor) that should
        be implemented in the base class.

        :param factor: The factor to be divided by.
        :return: The resulting quotient
        """


    @abstractmethod
    def reduce(self, vrs, values):
        """
        An abstract function for performing the observation (a.k.a conditioning) operation that should be implemented
        in the base class.

        :return: the resulting reduced factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param values: The values of vars.
        :return: The resulting reduced factor.
        """

    def observe(self, vrs, values):
        """
        (Alias for reduce) Reduce a factor by observing certain values for certain variables.

        :param vrs: The observed variables.
        :param values: The values of these variables.
        :return: The reduced factor.
        """
        return self.reduce(vrs, values)

    @abstractmethod
    def normalize(self):
        """
        An abstract function for performing factor normalization that should be implemented in the base class.

        :return: The normalized factor.
        """

    @abstractmethod
    def show(self):
        """
        An abstract function for printing the parameters of the factor that should be implemented in the base class.
        """

    @property
    def joint_distribution(self):
        """
        The joint distribution.
        """
        return self.copy()

    def __mul__(self, other):
        """
        Multiply this factor with another factor.

        :param other: The other factor.
        :return: The result.
        """
        return self.absorb(other)

    def __truediv__(self, other):
        """
        Divide this factor with another factor.

        :param other: The other factor.
        :return: The result.
        """
        return self.cancel(other)

    def __eq__(self, other):
        """
        Compare this factor to another factor.

        :param other: The other factor.
        :return: The equality result.
        """
        return self.equals(other)

    def __ne__(self, other):
        """
        Compare this factor to another factor.

        :param other: The other factor.
        :return: The inverse equality result.
        """
        return not self.__eq__(other)

