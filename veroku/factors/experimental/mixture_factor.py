"""
A module for instantiating and performing operations on multivariate Gaussian mixture distributions.
"""

# pylint: disable=cyclic-import
# pylint: disable=protected-access

# Standard imports
import cmath

# Third-party imports
import numpy as np
import numdifftools as nd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import special

# Local imports
from veroku.factors._factor import Factor
from veroku.factors import _factor_utils
import veroku.factors.gaussian as gauss
from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL

# TODO: confirm that the different GaussianMixture divide methods have sufficient stability and accuracy in their
#  approximations.
# TODO: Add tests for the divide methods.
# TODO: move to factors (non-experimental) once the divide methods have been checked and tested properly.


class MixtureFactor(Factor):
    """
    A Class for instantiating and performing operations on multivariate Gaussian mixture functions.
    """

    def __init__(self, factors):
        """
        A GaussianMixture constructor.

        :param factors: a list of Gaussian type objects with the same dimensionality.
        :type factors: Gaussian list
        :param cancel_method: The method of performing approximate division.
                              This is only applicable if the factor is a Gaussian mixture with more than one component.
                              0: moment match denominator to single Gaussian and divide
                              1: approximate modes of quotient as Gaussians
                              2: Use complex moment matching on inverse mixture sets (s_im) and approximate the inverse
                                 of each s_im as a Gaussian
        :type cancel_method: int
        """
        assert factors, "Error: empty list passed to constructor."
        self.components = [gaussian.copy() for gaussian in factors]
        self.num_components = len(factors)

        var_names0 = factors[0].var_names

        for component in self.components:
            if var_names0 != component.var_names:
                raise ValueError("inconsistent var_names in list of Gaussians.")
        super().__init__(var_names=var_names0)

    def equals(self, factor):
        """
        Check if this factor is the same as another factor.

        :param factor: The factor to compare with.
        :type factor: GaussianMixture
        :param float rtol: The absolute tolerance parameter (see numpy Notes for allclose function).
        :param float atol: The absolute tolerance parameter (see numpy Notes for allclose function).
        :return: Result of equals comparison between self and gaussian
        rtype: bool
        """
        # TODO: consider adding type error here rather
        if not isinstance(factor, self.__class__):
            raise TypeError(f"factor must be of GaussianMixture type but has type {type(factor)}")
        if factor.num_components != self.num_components:
            return False
        for i in range(self.num_components):
            found_corresponding_factor = False
            for j in range(i, self.num_components):
                if self.get_component(i).equals(factor.get_component(j)):
                    found_corresponding_factor = True
            if not found_corresponding_factor:
                return False
        return True

    def get_component(self, index):
        """
        Get the Gaussian component at an index.

        :param index: The index of the component to return.
        :type index: int
        :return: The component at the given index.
        :rtype: Gaussian
        """
        return self.components[index]

    def copy(self):
        """
        Make a copy of this Gaussian mixture.

        :return: The copied GaussianMixture.
        :rtype: GaussianMixture
        """
        component_copies = []
        for comp in self.components:
            component_copies.append(comp.copy())
        return self.__class__(component_copies)

    def multiply(self, factor):
        """
        Multiply this GaussianMixture with another factor.

        :param factor: The factor to multiply with.
        :type factor: Gaussian or Gaussian Mixture
        :return: The factor product.
        :rtype: GaussianMixture
        """
        new_components = []
        if isinstance(factor, gauss.Gaussian):
            for comp in self.components:
                new_components.append(comp.multiply(factor))
        elif isinstance(factor, self.__class__):
            for comp_ai in self.components:
                for comp_bi in factor.components:
                    new_components.append(comp_ai.multiply(comp_bi))
        else:
            raise TypeError("unsupported factor type.")
        return self.__class__(new_components)

    def divide(self, factor):
        """
        Divide this GaussianMixture by another factor.

        :param factor: The factor divide by.
        """
        raise NotImplementedError()

    def reduce(self, vrs, values):
        """
        Observe a subset of the variables in the scope of this Gaussian mixture and return the resulting factor.

        :param vrs: the names of the observed variable (list)
        :type vrs: str list
        :param values: the values of the observed variables (list or vector-like object)
        :type values: vector-like
        :return: the observation reduced factor.
        :rtype: GaussianMixture
        """
        new_components = []
        for comp in self.components:
            new_components.append(comp.reduce(vrs, values))
        return self.__class__(new_components)

    def marginalize(self, vrs, keep=True):
        """
        Integrate out variables from this Gaussian mixture.

        :param vrs: A subset of variables in the factor's scope.
        :type vrs: str list
        :param keep: Whether to keep or sum out vrs.
        :type keep: bool
        :return: the resulting marginal factor.
        :rtype: GaussianMixture
        """
        new_components = []
        for comp in self.components:
            new_components.append(comp.marginalize(vrs, keep))
        return self.__class__(new_components)

    def distance_from_vacuous(self):
        """
        NOTE: Not Implemented yet.
        Get the Kullback-Leibler (KL) divergence between the message factor and a vacuous (uniform) version of it.

        :return: The KL-divergence
        """
        raise NotImplementedError('This function has not been implemented yet.')

    def kl_divergence(self, factor):
        """
        NOTE: Not Implemented yet.
        Get the KL-divergence D_KL(self || factor) between a normalized version of this factor and another factor.

        :param factor: The other factor
        :type factor: GaussianMixture
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        raise NotImplementedError('This function has not been implemented yet.')

    def potential(self, x_val):
        """
        Get the value of the mixture potential at x_val.

        :param x_val: The point to evaluate the mixture at.
        :type x_val: vector-like
        :return: log of the value of the mixture potential at x_val.
        :rtype: float
        """
        total_potx = 0.0
        for comp in self.components:
            total_potx += comp.potential(x_val)
        return total_potx

    def _get_log_weights(self):
        """
        Get the log weights of the Gaussian mixture components.

        :return: the log weights
        :rtype: float
        """
        log_weights = []
        for comp in self.components:
            log_weights.append(comp.get_log_weight())
        return log_weights

    def _get_weights(self):
        """
        Get the weights of the Gaussian mixture components.

        :return: the log weights
        :rtype: float list
        """
        weights = []
        for comp in self.components:
            weights.append(comp.get_weight())
        return weights

    def get_log_weight(self):
        """
        Get the total log weight of the Gaussian mixture.

        :return: The log weight
        :rtype: float
        """
        return special.logsumexp(self._get_log_weights())

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        :rtype: GaussianMixture
        """
        unnormalized_log_weight = self.get_log_weight()
        new_components = []
        for comp in self.components:
            comp_copy = comp.copy()
            comp_copy._add_log_weight(-1.0 * unnormalized_log_weight)
            new_components.append(comp_copy)
        return self.__class__(new_components)

    @property
    def is_vacuous(self):
        """
        Check if a mixture distribution contains no information.

        :return: The result of the check.
        :rtype: Bool
        """
        # TODO: see how this is used. Should this be true if there is one vacuous component? Or only if all components
        #  are vacuous? Maybe make a contains vacuous function as well.
        for comp in self.components:
            if not comp._is_vacuous:
                return False
        return True

    def sample(self, num_samples):
        """
        Sample from this Gaussian mixture

        :param num_samples: The number of sample to draw.
        :type num_samples: int
        :return: The samples
        :rtype: float list
        """
        weights = self._get_weights()
        component_choice_samples = np.random.choice(
            range(len(weights)), size=num_samples, p=weights / np.sum(weights)
        )
        samples = []
        for comp_index in component_choice_samples:
            samples.append(self.components[comp_index].sample(1)[0])
        return np.array(samples)

    def plot(self, xlim, ylim, show_individual_components=False):
        """
        Plot the mixture potential function (only for 1d and 2d functions).

        :param log: if this is True, the log-potential will be plotted
        :type log: bool
        :param xlim: the x limits to plot the function over (optional)
        :type xlim: 2 element float list
        :param ylim: the y limits to plot the function over (optional and only used in 2d case)
        :type ylim: 2 element float list
        """
        if self.dim == 1:

            x_lower = xlim[0]
            x_upper = xlim[1]
            num_points = 200
            x_series = np.linspace(x_lower, x_upper, num_points)
            total_potx = np.zeros(num_points)
            for comp in self.components:
                potx = np.array([comp.potential(xi) for xi in x_series])
                if show_individual_components:
                    plt.plot(x_series, potx)
                total_potx += potx
            plt.plot(x_series, total_potx)
        elif self.dim == 2:
            self._plot_2d(log=False, xlim=xlim, ylim=ylim)
        else:
            raise NotImplementedError("Plotting not implemented for dim!=1.")

    def show(self):
        """
        Print the parameters of the Gaussian mixture distribution
        """
        for i, comp in enumerate(self.components):
            print("component ", i, "/", len(self.components))
            comp.show()

    def _plot_2d(self, xlim, ylim):
        """
        Plot a 2d Gaussian mixture potential function

        :param log: if this is True, the log-potential will be plotted
        :param xlim: the x limits to plot the function over (optional)
        :param ylim: the y limits to plot the function over (optional)
        """

        xlabel = self.var_names[0]
        ylabel = self.var_names[1]

        _factor_utils.plot_2d(func=self.potential, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
