"""
A module for instantiating and performing operations on multivariate Gaussian and Gaussian mixture distributions.
"""
# System imports
import cmath
import copy
import operator

# Third-party imports
import numpy as np
import numdifftools as nd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import special

# Local imports
from veroku.factors._factor import Factor
from veroku.factors import _factor_utils
from veroku.factors._factor_template import FactorTemplate
from veroku.factors.gaussian import Gaussian


class GaussianMixture(Factor):
    """
    A Class for instantiating and performing operations on multivariate Gaussian mixture functions.
    """

    def __init__(self, factors, cancel_method=0):
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
        assert factors, 'Error: empty list passed to constructor.'
        self.cancel_method = cancel_method
        self.components = [gaussian.copy() for gaussian in factors]
        self.num_components = len(factors)

        var_names0 = factors[0].var_names

        for component in self.components:
            if var_names0 != component.var_names:
                raise ValueError('inconsistent var_names in list of Gaussians.')
        super().__init__(var_names=var_names0)

    def equals(self, factor):
        """
        Check if this Gaussian mixture is the same as another factor.

        :param factor: The factor to compare to.
        :type factor: GaussianMixture
        :return: The result of the comparison.
        :rtype: bool
        """
        if not isinstance(factor, GaussianMixture):
            return False
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

        :param index: The index of teh component to return.
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
        for gauss in self.components:
            component_copies.append(gauss.copy())
        return GaussianMixture(component_copies)

    def multiply(self, factor):
        """
        Multiply this GaussianMixture with another factor.

        :param factor: The factor to multiply with.
        :type factor: Gaussian or Gaussian Mixture
        :return: The factor product.
        :rtype: GaussianMixture
        """
        new_componnents = []
        if isinstance(factor, Gaussian):
            for gauss in self.components:
                new_componnents.append(gauss.multiply(factor))
        elif isinstance(factor, GaussianMixture):
            for gauss_ai in self.components:
                for gauss_bi in factor.components:
                    new_componnents.append(gauss_ai.multiply(gauss_bi))
        else:
            raise TypeError('unsupported factor type.')
        return GaussianMixture(new_componnents)

    def divide(self, factor):
        """
        Divide this GaussianMixture by another factor.

        :param factor: The factor divide by.
        :type factor: Gaussian or Gaussian Mixture
        :return: The resulting factor quotient (approximate in the case of where both the numerator and denominator are
        GaussianMixtures with more than one component).
        :rtype: GaussianMixture
        """
        if isinstance(factor, Gaussian):
            single_gaussian = factor
        elif isinstance(factor, GaussianMixture):
            if factor.num_components == 1:
                single_gaussian = factor.get_component(index=0)
            else:
                if self.cancel_method == 0:
                    single_gaussian = factor.moment_match_to_single_gaussian()
                if self.cancel_method == 1:
                    return GaussianMixture._gm_division_m1(self, factor)
                if self.cancel_method == 2:
                    return GaussianMixture._gm_division_m2(self, factor)
        else:
            raise TypeError('unsupported factor type.')
        new_components = []
        for gauss in self.components:
            new_components.append(gauss.divide(single_gaussian))
        return GaussianMixture(new_components)

    def reduce(self, vrs, values):
        """
        Observe a subset of the variables in the scope of this Gaussian mixture and return the resulting factor.

        :param vrs: the names of the observed variable (list)
        :type vrs: str list
        :param values: the values of the observed variables (list or vector-like object)
        :type values: vactor-like
        :return: the observation reduced factor.
        :rtype: GaussianMixture
        """
        new_componnents = []
        for gauss in self.components:
            new_componnents.append(gauss.reduce(vrs, values))
        return GaussianMixture(new_componnents)

    def marginalize(self, vrs, keep=False):
        """
        Integrate out variables from this Gaussian mixture.

        :param vrs: A subset of variables in the factor's scope.
        :type vrs: str list
        :param keep: Whether to keep or sum out vrs.
        :type keep: bool
        :return: the resulting marginal factor.
        :rtype: GaussianMixture
        """
        new_componnents = []
        for gauss in self.components:
            new_componnents.append(gauss.marginalize(vrs, keep))
        return GaussianMixture(new_componnents)

    # pylint: disable=invalid-name
    def log_potential(self, x):
        """
        Get the log of the value of the Gaussian mixture potential at X.

        :param x: The point to evaluate the GaussianMixture at.
        :type x: vector-like
        :return: log of the value of the GaussianMixture potential at x.
        :rtype: float
        """
        log_potentials = []
        for gauss in self.components:
            log_potentials.append(gauss.log_potential(x))
        total_log_potx = special.logsumexp(log_potentials)
        return total_log_potx
    # pylint: enable=invalid-name

    def potential(self, x_val):
        """
        Get the value of the Gaussian mixture potential at x_val.

        :param x_val: The point to evaluate the GaussianMixture at.
        :type x_val: vector-like
        :return: log of the value of the GaussianMixture potential at x_val.
        :rtype: float
        """
        total_potx = 0.0
        for gauss in self.components:
            total_potx += gauss.potential(x_val)
        return total_potx

    def _get_means(self):
        """
        Get the means of the Gaussian components.

        :return: the mean vectors
        """
        means = []
        for gauss in self.components:
            means.append(gauss.get_mean())
        return means

    def _get_covs(self):
        """
        Get the covariance matrices of the Gaussian components.

        :return: the covariance matrices
        :rtype: np.ndarray list
        """
        covs = []
        for gauss in self.components:
            covs.append(gauss.get_cov())
        return covs

    def _get_log_weights(self):
        """
        Get the log weights of the Gaussian mixture components.

        :return: the log weights
        :rtype: float
        """
        log_weights = []
        for gauss in self.components:
            log_weights.append(gauss.get_log_weight())
        return log_weights

    def _get_weights(self):
        """
        Get the weights of the Gaussian mixture components.

        :return: the log weights
        :rtype: float list
        """
        weights = []
        for gauss in self.components:
            weights.append(gauss.get_weight())
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
        for gauss in self.components:
            gauss_copy = gauss.copy()
            gauss_copy._add_log_weight(-1.0*unnormalized_log_weight)
            new_components.append(gauss_copy)
        return GaussianMixture(new_components)

    @property
    def is_vacuous(self):
        """
        Check if a Gaussian mixture distribution contains no information. This is the case when the K matrices of all
        components are zero matrices.

        :return: The result of the check.
        :rtype: Bool
        """
        # TODO: see how this is used. Should this be true if there is one vacuous component? Or only if all components
        #  are vacuous? Maybe make a contains vacuous function as well.
        for gauss in self.components:
            if not gauss._is_vacuous:
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
        component_choice_samples = np.random.choice(range(len(weights)), size=num_samples,  p=weights / np.sum(weights))
        samples = []
        for comp_index in component_choice_samples:
            samples.append(self.components[comp_index].sample(1)[0])
        return np.array(samples)

    def _get_sensible_xlim(self):
        """
        Helper function for plot function to get the x limits that contain the majority of the Gaussian mass.

        :return: The x limits.
        :rtype: float list
        """
        x_lower_candidates = []
        x_upper_candidates = []
        for gauss in self.components:
            stddev = np.sqrt(gauss.get_cov()[0, 0])
            x_lower_candidates.append(gauss.get_mean()[0, 0] - 3 * stddev)
            x_upper_candidates.append(gauss.get_mean()[0, 0] + 3 * stddev)
        x_lower = min(x_lower_candidates)
        x_upper = max(x_upper_candidates)
        return [x_lower, x_upper]

    def plot(self, log=False, xlim=None, ylim=None, show_individual_components=False):
        """
        Plot the Gaussian mixture potential function (only for 1d and 2d functions).

        :param log: if this is True, the log-potential will be plotted
        :type log: bool
        :param xlim: the x limits to plot the function over (optional)
        :type xlim: 2 element float list
        :param ylim: the y limits to plot the function over (optional and only used in 2d case)
        :type ylim: 2 element float list
        """
        if self.dim == 1:
            if xlim is None:
                xlim = self._get_sensible_xlim()
            if xlim is not None:
                x_lower = xlim[0]
                x_upper = xlim[1]
            num_points = 200
            x_series = np.linspace(x_lower, x_upper, num_points)
            total_potx = np.zeros(num_points)
            for gauss in self.components:
                if log:
                    potx = np.array([gauss.log_potential(xi) for xi in x_series])
                else:
                    potx = np.array([gauss.potential(xi) for xi in x_series])
                if show_individual_components:
                    plt.plot(x_series, potx)
                total_potx += potx
            plt.plot(x_series, total_potx)
        elif self.dim == 2:
            self._plot_2d(log=log, xlim=xlim, ylim=ylim)
        else:
            raise NotImplementedError('Plotting not implemented for dim!=1.')

    def show(self):
        """
        Print the parameters of the Gaussian mixture distribution
        """
        for i, gauss in enumerate(self.components):
            print('component ', i, '/', len(self.components))
            gauss.show()

    # pylint: disable=invalid-name
    def moment_match_to_single_gaussian(self):
        """
        Calculate the mean and covariance of the Gaussian mixture and return a Gaussian with these parameters as an
        approximation of the Gaussian Mixture.

        :return: The Gaussian approximation.
        :rtype: Gaussian
        """
        new_mean = np.zeros([self.dim, 1])
        log_weights = []
        for gauss in self.components:
            wm = gauss.get_weight() * gauss.get_mean()
            new_mean += wm
            log_weights.append(gauss.get_log_weight())
        log_sum_weights = special.logsumexp(log_weights)
        sum_weights = np.exp(log_sum_weights)
        new_mean = new_mean / sum_weights
        new_cov = np.zeros([self.dim, self.dim])
        for gauss in self.components:
            ududT = gauss.get_weight() * (gauss.get_mean() - new_mean).dot((gauss.get_mean() - new_mean).transpose())
            new_cov += gauss.get_weight() * gauss.get_cov() + ududT
        new_cov = new_cov / sum_weights
        return Gaussian(cov=new_cov, mean=new_mean, log_weight=log_sum_weights,
                        var_names=self.components[0].var_names)
    # pylint: enable=invalid-name

    def argmax(self):
        """
        Find the input vector that maximises the potential function of the Gaussian mixture.

        :return: The argmax assignment.
        :rtype: numpy.ndarray
        """
        global_maximum_potential = -float('inf')
        global_argmax = None
        success = False

        def neg_gmm_pot(x_val):
            return -1.0 * self.potential(x_val)

        for gauss in self.components:
            res = minimize(neg_gmm_pot, x0=gauss.get_mean(), method='BFGS', options={'disp': False})
            x_local_max = res.x
            if res.success:
                success = True
                local_maximum_potential = self.potential(x_local_max)
                if local_maximum_potential > global_maximum_potential:
                    global_maximum_potential = local_maximum_potential
                    global_argmax = x_local_max
        if not success:
            raise Exception('could not find optimum')
        return global_argmax

    def _argmin(self):
        """
        Find the input vector that minimises the potential function of the Gaussian mixture with negative definite
        precision matrices.

        :return: The point where the function has a global minimum.
        :rtype: np.ndarray
        """
        global_minimum_potential = float('inf')
        global_argmin = None
        success = False

        def gmm_pot(x_vals):
            return self.potential(x_vals)

        for gauss in self.components:
            res = minimize(gmm_pot, x0=gauss.get_mean(), method='BFGS', options={'disp': False})
            x_local_min = res.x
            if res.success:
                success = True
                local_minimum_potential = self.potential(x_local_min)
                if local_minimum_potential < global_minimum_potential:
                    global_minimum_potential = local_minimum_potential
                    global_argmin = x_local_min
        if not success:
            raise Exception('could not find optimum')
        return global_argmin

    # pylint: disable=invalid-name
    def _moment_match_complex_gaussian(self):
        """
        Calculate the mean and covariance of the Gaussian mixture and return a Gaussian
        with these parameters a Gaussian approximation of the Gaussian Mixture.

        :return: The Gaussian approximation.
        :rtype: Gaussian
        """
        # TODO: check if the normal moment matching function can be replaced with this one.
        new_mean = np.zeros([self.dim, 1]).astype(complex)
        weights = []
        for gauss in self.components:
            c_weight = gauss.get_complex_weight()
            c_mean = np.linalg.inv(gauss.get_K()).dot(gauss.get_h()).astype(complex)
            new_mean += c_weight * c_mean
            weights.append(c_weight)
        sum_weights = sum(weights)
        new_mean = new_mean / sum_weights
        new_cov = np.zeros([self.dim, self.dim]).astype(complex)
        for gauss in self.components:
            c_weight = gauss.get_complex_weight()
            c_mean = np.linalg.inv(gauss.get_K()).dot(gauss.get_h()).astype(complex)
            ududT = c_weight * (c_mean - new_mean).dot((c_mean - new_mean).transpose())
            c_cov = np.linalg.inv(gauss.get_K()).astype(complex)
            new_cov += c_weight * c_cov + ududT
        new_cov = new_cov / sum_weights

        new_log_weight = np.log(sum_weights)

        new_K, new_h, new_g = GaussianMixture._complex_cov_form_to_real_canform(new_cov, new_mean, new_log_weight)
        return Gaussian(K=new_K, h=new_h, g=new_g,
                        var_names=self.components[0].var_names)
    # pylint: enable=invalid-name

    @staticmethod
    # pylint: disable=invalid-name
    def _complex_cov_form_to_real_canform(cov, mean, log_weight):
        """
        A helper function for _moment_match_complex_gaussian
        :param cov: the (possibly complex) covariance matrix
        :param mean: the (possibly complex) covariance matrix
        :param log_weight: the (possibly complex) log weight
        :return: the real parts of the converted canonical parameters (K, h, g)
        """
        K = np.linalg.inv(cov)
        h = K.dot(mean)
        uTKu = ((mean.transpose()).dot(K)).dot(mean)
        log_det_term = cmath.log(np.linalg.det(2.0 * np.pi * cov))
        g = abs(log_weight - 0.5 * uTKu - 0.5 * log_det_term)
        return K.real, h.real, g.real
    # pylint: enable=invalid-name

    @staticmethod
    # pylint: disable=invalid-name
    # pylint: disable=protected-access
    def _get_inverse_gaussians(gaussian_mixture_a, gaussian_mixture_b):
        """
        A helper function for _gm_division_m1. Returns the inverse components.
        For example:
        if gaussian_mixture_a = Ga1 + Ga2
        and gaussian_mixture_b = Gb1 + Gb2
        Then return Gb1/Ga1 + Gb1/Ga2 + Gb2/Ga1 + Gb2/Ga2
        :param gaussian_mixture_a: The numerator mixture
        :param gaussian_mixture_b: The denominator mixture
        :return: The inverse Gaussian components and the mode locations of the quotient:
                 gaussian_mixture_a/gaussian_mixture_b
        """
        minimum_locations = []
        for g_a in gaussian_mixture_a.components:
            inverse_components = []
            inverse_gaussian_mixtures = []
            for g_b in gaussian_mixture_b.components:
                inverse_components.append(g_b.divide(g_a))
            inverse_gaussian_mixture = GaussianMixture(inverse_components)
            inverse_gaussian_mixtures.append(inverse_gaussian_mixture)

            minimum_loc = inverse_gaussian_mixture._argmin()
            minimum_locations.append(minimum_loc)

        def neg_gm_log_quotient_potential(x):
            """
            Get the negative of the log value of the Gaussian mixture quotient (gma/gmb)
            """
            potential = -1.0 * (gaussian_mixture_a.log_potential(x) - gaussian_mixture_b.log_potential(x))
            return potential

        mode_locations = []
        for minimum_loc in minimum_locations:
            res = minimize(neg_gm_log_quotient_potential, x0=minimum_loc, method='BFGS', options={'disp': False})
            x_local_min = res.x
            if res.success:
                mode_locations.append(x_local_min)
        unique_mode_locations = _factor_utils.remove_duplicate_values(mode_locations, tol=1e-3)
        return inverse_gaussian_mixtures, unique_mode_locations
    # pylint: enable=protected-access
    # pylint: enable=invalid-name

    @staticmethod
    # pylint: disable=invalid-name
    # pylint: disable=protected-access
    def _gm_division_m1(gaussian_mixture_a, gaussian_mixture_b):
        """
        Method 1 for approximating the quotient (gma/gmb) of two Gaussian mixtures as a Gaussian mixture.
        This is an implementation of the method described in 'A Probabilistic Graphical Model Approach to Multiple Object Tracking' (section 8.2)
        The Gaussian mixture quotient is optimised from a set of starting points in order to find modes. Laplaces method is then used at these modes
        to appromate the mode as a Gaussian. The quotient is then approximated as a sum of these Gaussians.

        :params gma: The numerator Gaussian mixture
        :params gmb: The denominator Gaussian mixture
        returns: an approximation of the quotient function as a Gaussian mixture
        """

        resulting_gaussian_components = []
        inverse_gaussian_mixtures, unique_mode_locations = GaussianMixture._get_inverse_gaussians(gaussian_mixture_a,
                                                                                                  gaussian_mixture_b)
        inv_gm = GaussianMixture(inverse_gaussian_mixtures)

        for mode_location in unique_mode_locations:
            K_i = nd.Hessian(inv_gm.log_potential)(mode_location)
            mean_i = _factor_utils.make_column_vector(mode_location)
            h_i = K_i.dot(mean_i)

            uTKu = ((mean_i.transpose()).dot(K_i)).dot(mean_i)
            log_weight_i = -0.5 * np.log(np.linalg.det(K_i / (2.0 * np.pi))) * inv_gm.log_potential(mean_i)
            g_i = log_weight_i - 0.5 * uTKu + 0.5 * np.log(np.linalg.det(K_i / (2.0 * np.pi)))

            component = Gaussian(K=K_i, h=h_i, g=g_i,
                                 var_names=gaussian_mixture_a.components[0].var_names)
            resulting_gaussian_components.append(component)
        return GaussianMixture(resulting_gaussian_components)
    # pylint: enable=protected-access
    # pylint: enable=invalid-name

    @staticmethod
    # pylint: disable=protected-access
    def _gm_division_m2(gma, gmb):
        """
        Method 2 for approximating the quotient (gma/gmb) of two Gaussian mixtures as a Gaussian mixture.
        This function does applies the moment matching equations to sets of negative definite mixtures (using
        _moment_match_complex_gaussian). For each of these mixtures, a single negative definite 'Gaussian' (h(x)) is then
        obtained. This function is then inverted (g(x) = 1/h(x)), resulting in a positive definite Gaussian function.
        The Gaussian mixture quotient (gma/gmb) is then approximated as the sum of the g(x) functions.

        This method is can be much faster than _gm_division_m1 (~10 times [tested on for one dimensional functions]),
        but also results a Gaussian mixture with the same number of components as gma. Whereas _gm_division_m2 should
        have the same number of components as the modes of the quotient function.

        :params gma: The numerator Gaussian mixture
        :params gmb: The denominator Gaussian mixture
        returns: an approximation of the quotient function as a Gaussian mixture
        """
        # TODO: add check for dimensions that are not devided by
        #       - the variances in these will not change (if this makes sense)

        inverse_gaussian_mixtures = []
        resulting_gaussian_components = []
        count = 0
        for gaussian_a in gma.components:
            inverse_components = []
            for gaussian_b in gmb.components:
                conditional = gaussian_b.divide(gaussian_a)
                inverse_components.append(conditional)
            inverse_gaussian_mixture = GaussianMixture(inverse_components)
            inverse_gaussian_mixtures.append(inverse_gaussian_mixture)
            inverse_gaussian_approx = inverse_gaussian_mixture._moment_match_complex_gaussian()
            if _factor_utils.is_pos_def(inverse_gaussian_approx.K):
                count += 1
                # raise ValueError(' precision is negative definite.')
                print('Warning: precision is negative definite.')
            gaussian_approx = inverse_gaussian_approx._invert()
            resulting_gaussian_components.append(gaussian_approx)
        print('num pos def = ', count, '/', len(gma.components))
        resulting_gm = GaussianMixture(resulting_gaussian_components)
        return resulting_gm
    # pylint: enable=protected-access

    def _get_limits_for_2d_plot(self):  # pragma: no cover
        """
        Get x and y limits which includes most of the Gaussian mixture's mass, by considering
        the mean and variance of each Gaussian component.
        """
        x_lower_candidates = []
        x_upper_candidates = []
        y_lower_candidates = []
        y_upper_candidates = []
        for gauss in self.components:
            stddev_x = np.sqrt(gauss.get_cov()[0, 0])
            stddev_y = np.sqrt(gauss.get_cov()[1, 1])
            mean_x = gauss.get_mean()[0, 0]
            mean_y = gauss.get_mean()[1, 0]
            x_lower_candidates.append(mean_x - 3.0 * stddev_x)
            x_upper_candidates.append(mean_x + 3.0 * stddev_x)
            y_lower_candidates.append(mean_y - 3.0 * stddev_y)
            y_upper_candidates.append(mean_y + 3.0 * stddev_y)
        x_lower = min(x_lower_candidates)
        x_upper = max(x_upper_candidates)
        y_lower = min(y_lower_candidates)
        y_upper = max(y_upper_candidates)
        return [x_lower, x_upper], [y_lower, y_upper]

    def _plot_2d(self, log, xlim, ylim):  # pragma: no cover
        """
        Plot a 2d Gaussian mixture potential function
        :param log: if this is True, the log-potential will be plotted
        :param xlim: the x limits to plot the function over (optional)
        :param ylim: the y limits to plot the function over (optional)
        """
        if xlim is None and ylim is None:
            xlim, ylim = self._get_limits_for_2d_plot()
        elif xlim is not None and ylim is not None:
            pass
        else:
            print('Warning: partial limits received. Generating limits automatically.')
            xlim, ylim = self._get_limits_for_2d_plot()

        xlabel = self.var_names[0]
        ylabel = self.var_names[1]

        if not log:
            _factor_utils.plot_2d(func=self.potential, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        else:
            _factor_utils.plot_2d(func=self.log_potential, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)