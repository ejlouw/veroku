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
from scipy.optimize import minimize
import scipy.special

# Local imports
from veroku.factors._factor import Factor
from veroku.factors import _factor_utils


def make_random_gaussian(var_names, mean_range=[-10, 10], cov_range=[1, 10]):
    """
    Make a d dimensional random Gaussian by independently sampling the mean and covariance parameters from uniform
    distributions.

    :param var_names: The variable name of the factor
    :type var_names: str list
    :param mean_range: The range between which a mean will the uniformly sampled.
    :type mean_range: float list
    :param cov_range:The range between which a covariance will the uniformly sampled.
    :type cov_range: float list
    :return: The random Gaussian.
    :rtype: Gaussian
    """
    assert var_names, 'Error: var_names list cannot be empty.'
    dim = len(var_names)
    cov = np.random.uniform(cov_range[0], cov_range[1], [dim, dim]) * np.eye(dim, dim)
    mean = np.random.uniform(mean_range[0], mean_range[1], [dim, 1])
    random_gaussian = Gaussian(cov=cov, mean=mean, log_weight=0.0, var_names=var_names)
    return random_gaussian


def make_std_gaussian(var_names):
    """
    Make a d dimensional standard Gaussian.

    :param var_names: The variable name of the factor.
    :type var_names: str list
    :return: The standard Gaussian
    :rtype: Gaussian
    """
    assert var_names, 'Error: var_names list cannot be empty.'
    dim = len(var_names)
    cov = np.eye(dim, dim)
    mean = np.zeros([dim, 1])
    random_gaussian = Gaussian(cov=cov, mean=mean, log_weight=0.0, var_names=var_names)
    return random_gaussian


def split_gaussian(gaussian):
    """
    Split a Gaussian into a three component Gaussian Mixture, with different means.

    :param gaussian: The Gaussian distribution to split.
    :type gaussian: Gaussian
    :return: The split Gaussian Mixture.
    :rtype: GaussianMixture
    """
    if gaussian.dim != 1:
        raise NotImplementedError('Gaussian must be one dimensional.')
    weights = [1/3, 1/3, 1/3]
    full_cov = gaussian.get_cov()
    full_mean = gaussian.get_mean()

    covs = [w*full_cov for w in weights]
    means = [full_mean - np.sqrt(full_cov), full_mean, full_mean + np.sqrt(full_cov)]

    gaussians = []
    for mean, cov, weight in zip(means, covs, weights):
        gaussians.append(Gaussian(cov=cov,
                                  mean=mean,
                                  log_weight=np.log(weight),
                                  var_names=gaussian.var_names))
    gm = GaussianMixture(gaussians)
    return gm


class Gaussian(Factor):
    """
    A class for instantiating and performing operations om multivariate Gaussian distributions.
    """

    # pylint: disable=invalid-name
    def __init__(self, cov=None, mean=None, log_weight=None, K=None, h=None, g=None, var_names=None):
        """
        General constructor that can use either covariance form or canonical form parameters to construct a
        d-dimensional multivariate Gaussian object.

        :param cov: The covariance matrix (or variance scalar in 1-dimensional case).
        :type cov: dxd numpy.ndarray or float
        :param mean: The mean vector (or mean scalar in 1-dimensional case).
        :type mean: dx1 numpy.ndarray or float
        :param log_weight: the log of the weight (the Gaussian function integrates to the weight value)
        :type log_weight: float
        :param K: the precision matrix (or precision scalar in 1-dimensional case).
        :type K: dxd numpy.ndarray or float
        :param h: the h vector  (or h scalar in 1-dimensional case).
        :type h: dx1 numpy.ndarray or float
        :param g: the g parameter
        :type g: float
        :param var_names: a list of variable names where the order of the names correspond to the order in the
                         mean/covariance or K/h parameters
        :type var_names: str list
        """

        super().__init__(var_names=var_names)
        self.is_vacuous = False
        if all(v is None for v in [K, h, g]):
            if any(v is None for v in [cov, mean, log_weight]):
                raise ValueError('incomplete parameters')
            self.cov = _factor_utils.make_square_matrix(cov)
            self.mean = _factor_utils.make_column_vector(mean)
            assert len(self.cov.shape) == 2, "Error: Covariance matrix must be two dimensional."

            assert self.cov.shape[0] == self.dim, "Error: Covariance matrix dimension inconsistency."
            assert self.mean.shape[0] == self.dim, "Error: Mean vector dimension inconsistency."
            self.log_weight = _factor_utils.make_scalar(log_weight)
            self.K, self.h, self.g = None, None, None
            self.CANFORM = False
            self.COVFORM = True
        else:
            if any(v is None for v in [K, h, g]):
                raise ValueError('incomplete parameters')
            self.K = _factor_utils.make_square_matrix(K)
            self.h = _factor_utils.make_column_vector(h)
            if self.K.shape[0] != self.dim:
                pass
            assert len(self.K.shape) == 2, "Error: Precision matrix must be two dimensional."
            assert self.K.shape[0] == self.dim, "Error: Precision matrix dimension inconsistency."
            assert self.h.shape[
                       0] == self.dim, f"Error: h vector dimension inconsistency: {h.shape[0]} (should be {self.dim})"
            self.g = _factor_utils.make_scalar(g)
            self.cov, self.mean, self.log_weight = None, None, None
            self.CANFORM = True
            self.COVFORM = False

            # Note: this is important to allow vacuous Gaussians (that arise, for exmaple, when Gaussians are divided by
            # identical distributions or with unconditioned nonlinear Gaussians) to be merginalised.
            if np.allclose(self.K, np.zeros([self._dim, self._dim]), atol=1e-10):
                self.is_vacuous = True

    # pylint: enable=invalid-name

    @classmethod
    def make_vacuous(cls, var_names, g=0.0):
        """
        Make an vacuous Gaussian distribution with a zero precision matrix and zero h vector and zero g value.
        This 'Gaussian' is effectively a constant function (it has infinite variance and infinite mass) with value
        exp(g).

        :param var_names: The variable names.
        :type var_names: str list
        :param g: The g parameter.
        :type g: float
        :return: The vacuous Gaussian with zero K and h parameters.
        :rtype: Gaussian
        """
        dim = len(var_names)
        K_zero = np.zeros([dim, dim])
        h_zero = np.zeros([dim, 1])
        g_zero = g
        return cls(K=K_zero, h=h_zero, g=g_zero, var_names=var_names)

    # pylint: disable=invalid-name
    # pylint: disable=protected-access
    def _reorder_parameters(self, new_order_vars):
        """
        Reorder the values in the matrix and vector parameters according to the order of new_order_vars

        :param new_order_vars: The same variable names as in self.var_names, in a different order
        :type new_order_vars: str list
        """
        if new_order_vars == self._var_names:
            return
        assert set(new_order_vars) == set(self._var_names), \
            'Error: new_order_vars must contain the same variable names as in var_names.'
        new_order = [new_order_vars.index(var) for var in self._var_names]

        if self.CANFORM:
            new_K = np.zeros([self.dim, self.dim])
            new_h = np.zeros([self.dim, 1])
            for i, new_i in enumerate(new_order):
                new_h[new_i, 0] = self.h[i, 0]
                for j, new_j in enumerate(new_order):
                    new_K[new_i, new_j] = self.K[i, j]
            self.K = new_K
            self.h = new_h

        if self.COVFORM:
            new_cov = np.zeros([self.dim, self.dim])
            new_mean = np.zeros([self.dim, 1])
            for i, new_i in enumerate(new_order):
                new_mean[new_i, 0] = self.mean[i, 0]
                for j, new_j in enumerate(new_order):
                    new_cov[new_i, new_j] = self.cov[i, j]
            self.cov = new_cov
            self.mean = new_mean
        self._var_names = new_order_vars

    # pylint: enable=protected-access
    # pylint: enable=invalid-name

    # pylint: disable=protected-access
    def _canform_equals(self, gaussian, rtol, atol):
        """
        Helper function for check equivalence of canonical form parameters.
        """
        if not np.isclose(self.g, gaussian.get_g(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        gaussian._reorder_parameters(self.var_names)
        if not np.allclose(self.h, gaussian.get_h(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        if not np.allclose(self.K, gaussian.get_K(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        return True
    # pylint: enable=protected-access

    # pylint: disable=protected-access
    def _covform_equals(self, gaussian, rtol, atol):
        """
        Helper function for check equivalence of covariance form parameters.
        """
        if not np.isclose(self.log_weight, gaussian.get_log_weight(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        gaussian._reorder_parameters(self.var_names)
        if not np.allclose(self.mean, gaussian.get_mean(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        if not np.allclose(self.cov, gaussian.get_cov(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        return True
    # pylint: enable=protected-access

    def equals(self, factor, rtol=1e-05, atol=1e-05):
        """
        Check if this factor is the same as another factor.

        :param factor: The factor to compare with.
        :type factor: Gaussian
        :param rtol: The absolute tolerance parameter (see numpy Notes for allclose function).
        :type rtol: float
        :param atol: The absolute tolerance parameter (see numpy Notes for allclose function).
        :type atol: float
        :return: Result of equals comparison between self and gaussian
        rtype: bool
        """
        # if not isinstance(factor, Gaussian):
        #    return False
        if set(self._var_names) != set(factor.var_names):
            return False

        gaussian_copy = factor.copy()
        # pylint: disable=protected-access
        if self._var_names != factor.var_names:
            gaussian_copy._reorder_parameters(self._var_names)
        if self.CANFORM:
            if not self._canform_equals(factor, rtol, atol):
                return False
        if self.COVFORM:
            if not self._covform_equals(factor, rtol, atol):
                return False
        # pylint: enable=protected-access
        return True

    # pylint: disable=invalid-name
    def get_K(self):
        """
        Get the K parameter.

        :return: The K parameter.
        :type K: dxd numpy.ndarray or float
        """
        self._update_canform()
        if self.K is not None:
            return self.K.copy()
        return None

    # pylint: enable=invalid-name

    # pylint: disable=invalid-name
    def get_h(self):
        """
        Get the h vector.

        :return: The h parameter.
        :rtype: float
        """
        self._update_canform()
        if self.h is not None:
            return self.h.copy()
        return None
    # pylint: enable=invalid-name

    # pylint: disable=invalid-name
    def get_g(self):
        """
        Get the g parameter.

        :return: The g parameter.
        :rtype: float
        """
        self._update_canform()
        return self.g
    # pylint: enable=invalid-name

    def cov_exists(self):
        if self.COVFORM:
            return True
        else:
            try:
                np.linalg.inv(self.K)
            except np.linalg.LinAlgError:
                return False
        return True

    # pylint: disable=invalid-name
    def get_cov(self):
        """
        Get the covariance parameter.

        :return: The cov parameter.
        :rtype: numpy.ndarray
        """
        self._update_covform()
        if self.cov is not None:
            return self.cov.copy()
        return None
    # pylint: enable=invalid-name

    def get_mean(self):
        """
        Get the mean parameter.

        :return: The mean parameter.
        :rtype: numpy.ndarray
        """
        self._update_covform()
        if self.mean is not None:
            return self.mean.copy()
        return None

    def get_log_weight(self):
        """
        Get the log weight parameter.
        :return: The log_weight parameter.
        :rtype: numpy.ndarray
        """
        self._update_covform()
        return self.log_weight

    def normalise(self):
        gaussian_copy = self.copy()
        gaussian_copy._update_covform()
        gaussian_copy.log_weight = 0.0
        return gaussian_copy

    def get_complex_weight(self):
        """
        Get (possibly complex) weight parameter.

        :return: The weight parameter.
        :rtype: float
        """
        return cmath.exp(self._get_complex_log_weight())

    def _get_complex_log_weight(self):
        """
        Get the log weight even in cases where the determinant of the covariance matrix is negative. In such cases the
        log_weight no longer corresponds to the integral and the log of the weight will have a imaginary component.
        Computing the complex log-weight can however still be useful: it is used, for example, in the (experimental)
        Gaussian mixture division function (_gm_division_m2).
        """
        # pylint: disable=invalid-name
        self._update_canform()
        cov = np.linalg.inv(self.K)
        mean = cov.dot(self.h)
        uTKu = ((mean.transpose()).dot(self.K)).dot(mean).astype(complex)
        log_weight = self.g + 0.5 * uTKu + 0.5 * cmath.log(np.linalg.det(2.0 * np.pi * cov))
        return log_weight
        # pylint: enable=invalid-name

    def _invert(self):
        """
        Invert this Gaussian (1/Gaussian). This is used in the approximate Gaussian mixture division algorithms.

        :return: The inverted Gaussian.
        """
        assert self.CANFORM
        gaussian_copy = self.copy()
        gaussian_copy.K = (-1.0) * self.K
        gaussian_copy.h = (-1.0) * self.h
        gaussian_copy.g = (-1.0) * self.g
        return gaussian_copy

    def _subtract_log_weight(self, log_weight_to_subtract):
        """
        Subtract log value to the log weight.

        :param log_weight_to_subtract: The log value to subtract from the weight.
        :type log_weight_to_subtract: float
        """
        # TODO: replace with _add_log_weight
        self._update_covform()
        self.log_weight -= log_weight_to_subtract

    def _add_log_weight(self, log_weight_to_add):
        """
        Add log value to the log weight.

        :param log_weight_to_add: The log value to add to the weight.
        :type log_weight_to_add: float
        """
        if self.CANFORM:
            self.g += log_weight_to_add
        if self.COVFORM:
            self.log_weight += log_weight_to_add

    def get_weight(self):
        """
        Get the weight of the distribution - the value of the integral of the (possibly unnormalised) distribution.

        :return: The weight.
        :rtype: float
        """
        return np.exp(self.get_log_weight())

    def _fix_non_psd_matrices(self):
        if self.COVFORM:
            if not _factor_utils.is_pos_def(self.cov):
                self.cov = _factor_utils.get_closest_psd_from_symmetric_matrix(self.cov)
        if self.CANFORM:
            if not _factor_utils.is_pos_def(self.K):
                self.K = _factor_utils.get_closest_psd_from_symmetric_matrix(self.K)

    def marginalise(self, vrs, keep=False):
        """
        Integrate out variables from this Gaussian.

        :param vrs: A subset of variables in the factor's scope.
        :type vrs: str list
        :param keep: Whether to keep or sum out vrs.
        :type keep: bool
        :return: The resulting Gaussian marginal.
        :rtype: Gaussian
        """
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_integrate_out = list(set(self.var_names) - set(vars_to_keep))
        if self.is_vacuous:
            # TODO: check this (esp log_weight)
            # print('Warning: marginalising vacuous Gaussian')
            return Gaussian.make_vacuous(var_names=vars_to_keep)

        if self.CANFORM:
            xx_indcs = [self.var_names.index(var_x) for var_x in vars_to_keep]
            yy_indcs = [self.var_names.index(var_y) for var_y in vars_to_integrate_out]
            Kxx = self.K[np.ix_(xx_indcs, xx_indcs)]
            Kxy = self.K[np.ix_(xx_indcs, yy_indcs)]
            Kyx = Kxy.transpose()
            Kyy = self.K[yy_indcs][:, yy_indcs]
            try:
                inv_Kyy = np.linalg.inv(Kyy)
            except np.linalg.LinAlgError as e:
                print('vars_to_integrate_out = ', vars_to_integrate_out)
                self.show_vis(figsize=(20, 16))
                # return Gaussian.make_vacuous(var_names=vars_to_keep)
                raise e
            #    # TODO: check this (not sure if it is correct)
            #    return self.make_vacuous(vars_to_keep, g=self.g)
            hx = self.h[xx_indcs]
            hy = self.h[yy_indcs]
            Kxy_inv_Kyy = Kxy.dot(np.linalg.inv(Kyy))
            K = Kxx - Kxy_inv_Kyy.dot(Kyx)
            h = hx - Kxy_inv_Kyy.dot(hy)
            g = self.g + 0.5 * ((hy.T.dot(np.linalg.inv(Kyy)).dot(hy)) + np.log(np.linalg.det(2.0 * np.pi * inv_Kyy)))
            return Gaussian(K=K, h=h, g=g, var_names=vars_to_keep)

        self._update_covform()
        indices_to_keep = [self._var_names.index(variable) for variable in vars_to_keep]
        marginal_var_names = vars_to_keep.copy()

        if self.is_vacuous:
            raise ValueError('cannot marginalise vacuous Gaussian.')

        marginal_cov = self.cov[np.ix_(indices_to_keep, indices_to_keep)]
        marginal_mean = self.mean[np.ix_(indices_to_keep, [0])]
        return Gaussian(cov=marginal_cov, mean=marginal_mean, log_weight=self.log_weight,
                        var_names=marginal_var_names)

    def _update_canform(self):
        """
        Update the canonical form parameters of the Gaussian.
        """
        # pylint: disable=invalid-name
        if self.CANFORM:
            return
        assert self.COVFORM
        try:
            self.K = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError as e:
            print('self.cov = ', self.cov)
            raise e
        self.h = self.K.dot(self.mean)
        uTKu = ((self.mean.transpose()).dot(self.K)).dot(self.mean)

        try:
            det2picov = np.linalg.det(2.0 * np.pi * self.cov)
            if det2picov < 0.0:
                raise ValueError(f'invalid value in log: det2picov={det2picov} ')
            g_ = self.log_weight - 0.5 * uTKu - 0.5 * np.log(det2picov)
        except Exception as e:
            print(self.cov)
            raise e
        self.g = _factor_utils.make_scalar(g_)
        self.CANFORM = True
        # pylint: enable=invalid-name

    def _update_covform(self):
        """
        Update the covariance form parameters of the Gaussian.
        """
        if self.is_vacuous:
            raise Exception('cannot update covariance form for vacuous Gaussian.')
        # pylint: disable=invalid-name
        if self.COVFORM:
            return
        assert self.CANFORM
        try:
            self.cov = np.linalg.inv(self.K)
        except np.linalg.LinAlgError as e:
            np.set_printoptions(edgeitems=20, precision=2, linewidth=500)
            print('cannot update cov form: self.K \n= ', self.var_names, '\n', self.K)
            # self.show_vis()
            print(e)
            raise e
        assert not np.isnan(np.sum(self.cov)), 'Error: nan values in cov matrix after inversion.'
        self.mean = self.cov.dot(self.h)
        uTKu = ((self.mean.transpose()).dot(self.K)).dot(self.mean)
        det_2_pi_cov = np.linalg.det(2.0 * np.pi * self.cov)
        # if det_2_pi_cov < 0:
        #    print('Error: 0 > det_2_pi_cov = ', det_2_pi_cov)
        log_weight_ = self.g + 0.5 * uTKu + 0.5 * np.log(det_2_pi_cov)
        self.log_weight = _factor_utils.make_scalar(log_weight_)
        self.COVFORM = True
        # pylint: enable=invalid-name

    def absorb(self, factor):
        """
        Multiply this Gaussian with another factor.
        :param factor: the factor to multiply with
        :return: the resulting factor
        """
        # if isinstance(factor, NonLinearGaussian):
        #    return factor.absorb(self)
        return self._absorb_or_cancel(factor, operator.add)

    def cancel(self, factor):
        """
        Divide this Gaussian by another facotr.
        :param factor: the factor to divide by
        :return: the resulting factor
        """
        return self._absorb_or_cancel(factor, operator.sub)

    def _absorb_or_cancel(self, factor, operator_function):
        """
        A general function which can either perform Gaussian multiplication or division (which are very similar).
        :param factor: the gaussian to multiply or divide by
        :param operator_function: the operator to use one the Gaussian canonical parameters
                        ('+' for multiplication and '-' for division)
        :return: the resulting GaussianMixture
        """
        # pylint: disable=invalid-name
        K_a = self.get_K()
        K_b = factor.get_K()
        assert len(K_a.shape) == 2
        assert len(K_b.shape) == 2
        K_c, new_vars_0 = _factor_utils.indexed_square_matrix_operation(K_a, K_b, self._var_names,
                                                                       factor.var_names, operator_function)
        g_a = self.get_g()
        g_b = factor.get_g()
        g_c = operator_function(g_a, g_b)

        h_a = self.get_h()
        h_b = factor.get_h()
        h_c, new_vars_1 = _factor_utils.indexed_column_vector_operation(h_a, h_b, self._var_names,
                                                                       factor.var_names, operator_function)
        assert new_vars_0 == new_vars_1
        return Gaussian(K=K_c, h=h_c, g=g_c, var_names=new_vars_0)
        # pylint: enable=invalid-name

    def sample(self, num_samples):
        """
        Draw samples from the Gaussian distribution.
        :param num_samples: The number of samples to draw.
        :type num_samples:
        :return: The samples.
        :rtype: int
        """
        std_gaussian_samples = np.random.normal(0, 1, [self.dim, num_samples])
        Xs = std_gaussian_samples
        self._update_covform()
        L = np.linalg.cholesky(self.cov)
        X = L.dot(Xs) + self.mean
        return X

    # pylint: disable=invalid-name
    def _get_observation_reduced_canonical_vars(self, observed_indices, unobserved_indices, observed_vec):
        """
        A helper function for observe.
        :param observed_indices: The observed variable indices
        :type observed_indices: int list
        :param unobserved_indices: The unobserved variable indices
        :type unobserved_indices: int list
        :return: the reduced parameters K_observed, h_observed, g_observed
        :rtype: numpy.ndarray, numpy.ndarray, float
        """
        K = self.get_K()
        h = self.get_h()
        K_reduced = 0
        h_reduced = 0
        if unobserved_indices:
            K_XX = K[np.ix_(unobserved_indices, unobserved_indices)]
            K_reduced = K_XX.copy()
            K_XY = K[np.ix_(unobserved_indices, observed_indices)]
            h_X = h[np.ix_(unobserved_indices, [0])]
            h_reduced = h_X - K_XY.dot(observed_vec)

        K_YY = K[np.ix_(observed_indices, observed_indices)]
        h_Y = h[np.ix_(observed_indices, [0])]
        g_reduced = self.get_g() + (h_Y.transpose()).dot(observed_vec) - 0.5 * (
            (observed_vec.transpose()).dot(K_YY)).dot(observed_vec)

        return K_reduced, h_reduced, g_reduced
    # pylint: enable=invalid-name

    # pylint: disable=invalid-name
    def observe(self, vrs, values):
        """
        Observe a subset of the variables in the scope of this Gaussian and return the resulting factor.

        :param vrs: the names of the observed variable (list)
        :type vrs: str list
        :param values: the values of the observed variables
        :type values: vector-like
        :return: the resulting Gaussian
        :rtype: Gaussian
        """

        observed_vec = _factor_utils.make_column_vector(values)

        assert isinstance(vrs, list)  # just to future-proof interface
        assert set(vrs) <= set(self._var_names), \
            'Error: observed variables must be a subset of the gaussian variables.'

        unobserved_vars = list(set(self._var_names) - set(vrs))
        unobserved_vars.sort()  # the above operations seems to return inconsistent orderings
        observed_indices = [self._var_names.index(v) for v in vrs]
        unobserved_indices = [self._var_names.index(v) for v in unobserved_vars]

        K_reduced, h_reduced, g_reduced = self._get_observation_reduced_canonical_vars(
            observed_indices=observed_indices,
            unobserved_indices=unobserved_indices,
            observed_vec=observed_vec)

        return Gaussian(K=K_reduced, h=h_reduced, g=g_reduced, var_names=unobserved_vars)
        # pylint: enable=invalid-name

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        if self.is_vacuous:
            return 0.0
        else:
            return float('inf')

    def kl_divergence(self, factor, normalise_factor=True):
        """
        Get the KL-divergence D_KL(self||factor) between a normalised version of this factor and another factor.
        Reference https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf, page 1.

        :param factor: The other factor
        :type factor: Gaussian
        :param normalise_factor: Whether or not to normalise the other factor before computing the KL-divergence.
        :type normalise_factor: bool
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        if self.dim != factor.dim:
            raise ValueError('cannot get KL-divergence between Gaussians of different dimensionalities.')
        if self.is_vacuous and factor.is_vacuous:
            return 0.0
        if self.is_vacuous or factor.is_vacuous:
            return float('inf')

        if self.equals(factor):
            return 0.0
        normalised_self = self.normalise()
        factor_ = factor
        if normalise_factor:
            factor_ = factor.normalise()
        inv_cov_q = factor_.get_K()
        inv_cov_p = normalised_self.get_K()
        cov_p = normalised_self.get_cov()

        u_q = factor_.get_mean()
        u_p = normalised_self.get_mean()

        det_inv_cov_q = np.linalg.det(inv_cov_q)
        det_inv_cov_p = np.linalg.det(inv_cov_p)
        if det_inv_cov_q == 0.0:
            if det_inv_cov_p != 0.0:
                return np.inf
            else:
                det_term = 0.0
        else:
            det_term = 0.5 * cmath.log(det_inv_cov_p / det_inv_cov_q)
        trace_term = 0.5 * np.trace(inv_cov_q.dot(cov_p))
        mahalanobis_term = 0.5 * (u_p - u_q).T.dot(inv_cov_q).dot(u_p - u_q)
        dim_term = 0.5 * normalised_self.dim
        kld = det_term + trace_term + mahalanobis_term - dim_term
        # TODO:Add warning or error if this is negative and remove abs below
        return np.abs(kld[0][0])

    def copy(self):
        """
        Make a copy of this Gaussian.

        :return: The copied Gaussian.
        :rtype: Gaussian
        """
        if self.COVFORM:
            return Gaussian(cov=self.cov.copy(),
                            mean=self.mean.copy(),
                            log_weight=self.log_weight,
                            var_names=copy.deepcopy(self._var_names))
        if self.CANFORM:
            return Gaussian(K=self.K.copy(),
                            h=self.h.copy(),
                            g=self.g,
                            var_names=copy.deepcopy(self._var_names))
        raise Exception('Gaussian is neither in canonical form nor in covariance form?')

    def potential(self, x_val):
        """
        Get the value of the Gaussian potential at x_val.

        :param x_val: The vector (or vector-like object) to evaluate the Gaussian at
        :type x_val: numpy.ndarray
        :return: The value of the Gaussian potential at x_val.
        """
        return np.exp(self.log_potential(x_val))

    # pylint: disable=invalid-name
    def log_potential(self, x_val, vrs=None):
        """
        Get the log of the value of the Gaussian potential at x_val.

        :param x_val: the vector (or vector-like object) to evaluate the GaussianMixture at
        :type x_val: vector-like
        :param vrs: The variables corresponding to the values in x_val.
        :type vrs: str list
        :return: The log of the value of the GaussianMixture potential at x_val.
        """
        if vrs is not None:
            assert set(vrs) == set(self.var_names)
            if isinstance(x_val, np.ndarray):
                x_val = x_val.ravel()
            x_val_list = list(x_val)
            assert len(x_val_list) == len(self.var_names)
            x_val = [x_val_list[vrs.index(v)] for v in self.var_names]
        x_vec = _factor_utils.make_column_vector(x_val)
        if self.COVFORM:
            log_norm = self.log_weight - 0.5 * np.log(np.linalg.det(2.0 * np.pi * self.cov))
            K = np.linalg.inv(self.cov)
            exponent = ((-0.5 * (self.mean - x_vec).transpose()).dot(K)).dot(self.mean - x_vec)
            log_potx = log_norm + exponent

        if self.CANFORM:
            log_potx = -0.5 * (x_vec.transpose().dot(self.K)).dot(x_vec) + x_vec.transpose().dot(self.h) + self.g

        return log_potx[0, 0]

        raise Exception('Gaussian is neither in canonical form nor in covariance form?')

    # pylint: enable=invalid-name

    def show(self, update_covform=True, show_canform=False):  # pragma: no cover
        """
        Print the parameters of the Gaussian distribution

        :param update_covform: Whether or not to update the covariance form before showing.
        :type update_covform: bool
        :param show_canform: Whether or not toshow the canonical form as well.
        :type show_canform: bool
        """
        np.set_printoptions(edgeitems=3)
        np.set_printoptions(precision=4)
        np.core.arrayprint._line_width = 200
        self_copy = self.copy()
        if not self.is_vacuous and update_covform:
            self_copy._update_covform()
        print('vars = ', self_copy.var_names)
        if self_copy.COVFORM:
            print('Cov = \n', self_copy.cov)
            print('mean = \n', self_copy.mean)
            print('log_weight = \n', self_copy.log_weight)
            return
        if self_copy.CANFORM and show_canform:
            print('K = \n', self_copy.K)
            print('h = \n', self_copy.h)
            print('g = \n', self_copy.g)
        print('is_vacuous: ', self_copy.is_vacuous)

    def show_vis(self, figsize=(10, 8)):
        """
        Visualise the parameters with plots.

        :param figsize: The figure size.
        :type figsize: 2 element int tuple
        """
        # TODO: find a way of making cov matrix square.
        f, [ax_cov, ax_mean] = plt.subplots(nrows=2, figsize=figsize)  # ,sharex=True)
        cov_df = pd.DataFrame(data=self.get_cov(), index=self.var_names, columns=self.var_names)
        mean_df = pd.DataFrame(data=self.get_mean(), index=self.var_names, columns=['var_names'])
        sns.heatmap(cov_df, ax=ax_cov, cbar=True, cbar_kws=dict(use_gridspec=False, location="top"), annot=True)

        mean_df.plot.bar(ax=ax_mean, legend=False)
        plt.xticks(rotation=0)

    def _get_limits_for_2d_plot(self):  # pragma: no cover
        """
        Get x and y limits which includes most of the Gaussian mixture's mass, by considering
        the mean and variance of each Gaussian component.
        """
        self_copy = self.copy()
        self_copy._update_covform()
        x_lower = self_copy.mean[0, 0] - 3.0 * np.sqrt(self_copy.cov[0, 0])
        x_upper = self_copy.mean[0, 0] + 3.0 * np.sqrt(self_copy.cov[0, 0])
        y_lower = self_copy.mean[1, 0] - 3.0 * np.sqrt(self_copy.cov[1, 1])
        y_upper = self_copy.mean[1, 0] + 3.0 * np.sqrt(self_copy.cov[1, 1])
        return [x_lower, x_upper], [y_lower, y_upper]

    # TODO: reconcile with GaussianMixture _plot_2d
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

    def plot(self, log=False, xlim=None, ylim=None):
        """
        Plot the Gaussian mixture potential function (only for 1d and 2d functions)

        :param log: if this is True, the log-potential will be plotted
        :param xlim: the x limits to plot the function over (optional)
        :type xlim: 2 element float list
        :param ylim: the y limits to plot the function over (optional and only used in 2d case)
        :type ylim: 2 element float list
        """
        # TODO: Replace with gaussian mixture plot.
        if self.dim == 1:
            if xlim is None:
                stddev = np.sqrt(self.get_cov()[0, 0])
                lb = self.get_mean()[0, 0] - 3 * stddev
                ub = self.get_mean()[0, 0] + 3 * stddev
                xlim = [lb, ub]
            if xlim is not None:
                x_lower = xlim[0]
                x_upper = xlim[1]
            num_points = 200
            x_series = np.linspace(x_lower, x_upper, num_points)
            if log:
                potx = np.array([self.log_potential(xi) for xi in x_series])
            else:
                potx = np.array([self.potential(xi) for xi in x_series])
            plt.plot(x_series, potx)
        elif self.dim == 2:
            self._plot_2d(log=log, xlim=xlim, ylim=ylim)
        else:
            raise NotImplementedError('Plotting not implemented for dim!=1.')


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

    def absorb(self, factor):
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
                new_componnents.append(gauss.absorb(factor))
        elif isinstance(factor, GaussianMixture):
            for gauss_ai in self.components:
                for gauss_bi in factor.components:
                    new_componnents.append(gauss_ai.absorb(gauss_bi))
        else:
            raise TypeError('unsupported factor type.')
        return GaussianMixture(new_componnents)

    def cancel(self, factor):
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
            new_components.append(gauss.cancel(single_gaussian))
        return GaussianMixture(new_components)

    def observe(self, vrs, values):
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
            new_componnents.append(gauss.observe(vrs, values))
        return GaussianMixture(new_componnents)

    def marginalise(self, vrs, keep=False):
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
            new_componnents.append(gauss.marginalise(vrs, keep))
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
        total_log_potx = scipy.special.logsumexp(log_potentials)
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

    @property
    def is_vacuous(self):
        # TODO: see how this is used. Should this be true if there is one vacuous component? Or only if all components
        #  are vacuous? Maybe make a contains vacuous function as well.
        for gauss in self.components:
            if not gauss.is_vacuous:
                return False
        return True

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
        return scipy.misc.logsumexp(self._get_log_weights())

    def normalise(self):
        """
        Normalise the Gaussian mixture.

        :return: The normalised factor.
        :rtype: GaussianMixture
        """
        unnormalised_log_weight = self.get_log_weight()
        new_components = []
        for gauss in self.components:
            gauss_copy = gauss.copy()
            gauss_copy._subtract_log_weight(unnormalised_log_weight)
            new_components.append(gauss_copy)
        return GaussianMixture(new_components)

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
        log_sum_weights = scipy.misc.logsumexp(log_weights)
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
                inverse_components.append(g_b.cancel(g_a))
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
                conditional = gaussian_b.cancel(gaussian_a)
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
