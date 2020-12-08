"""
A module for instantiating and performing operations on multivariate Gaussian and Gaussian mixture distributions.
"""
# System imports
import cmath
import copy
import operator

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from veroku.factors._factor import Factor
from veroku.factors import _factor_utils
from veroku.factors._factor_template import FactorTemplate


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
    # TODO: Add non diagonal covariance matrix sampling and add test for PSD covariance matrices.
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


def make_linear_gaussian(A, N, conditioning_vars, conditional_vars):
    """
    Make a linear Gaussian factor.

    :param A: The linear transform matrix.
    :type A: Numpy array
    :param N: The additive noise covariance matrix.
    :type A: Numpy array
    :param conditioning_vars: The list of conditioning variables.
    :param conditional_vars: The list of conditional variables.
    :return: The linear Gaussian Template
    :rtype: Gaussian
    """

    # TODO: improve this by adding the correct equations for calculating the linear Gaussian parameters that do not rely
    #  on the dummy factor multiplication.
    X_dim = len(conditioning_vars)
    Sxx = np.eye(X_dim)
    Kxx = np.linalg.inv(Sxx)

    ux = np.zeros([X_dim, 1])
    hx = Kxx @ ux
    uxTKxxux = ux.transpose() @ hx
    gx = - 0.5 * uxTKxxux + np.log(1.0) - 0.5 * np.log(np.linalg.det(2.0 * np.pi * Sxx))

    S = np.block([[Sxx, Sxx @ A.T],
                  [A @ Sxx, A @ Sxx @ A.T + N]])
    K_joint = np.linalg.inv(S)
    K_cpd = K_joint.copy()
    K_cpd[:X_dim, :X_dim] = K_cpd[:X_dim, :X_dim] - Kxx

    mean_joint = np.block([[ux], [A.T @ ux]])

    h_joint = K_joint @ mean_joint
    h_cpd = h_joint.copy()
    h_cpd[:X_dim] = h_cpd[:X_dim] - hx

    uTKux = mean_joint.T @ K_joint @ mean_joint
    g_joint = 0.5 * uTKux + np.log(1.0) - 0.5 * np.log(np.linalg.det(2.0 * np.pi * S))
    g_cpd = g_joint - gx
    var_names = conditioning_vars + conditional_vars

    return Gaussian(var_names=var_names, K=K_cpd, h=h_cpd, g=g_cpd)


def make_linear_gaussian_cpd_template(A, N, conditioning_var_templates, conditional_var_templates):
    """
    Make a linear Gaussian factor template.

    :param A: The linear transform matrix.
    :type A: Numpy array
    :param N: The additive noise covariance matrix.
    :type A: Numpy array
    :param conditioning_var_templates: The list of formattable strings for the conditioning variables (i.e: ['var_a{i}_{t}', 'var_b{i}_{t}'])
    :param conditional_var_templates: The list of formattable strings for the conditional variables (i.e: ['var_c{i}_{t}', 'var_d{i}_{t}'])
    :return: The linear Gaussian Template
    :rtype: Gaussian
    """
    assert len(conditioning_var_templates) == A.shape[0]
    assert N.shape[0] == A.shape[0]
    nlg = make_linear_gaussian(A, N, conditioning_var_templates, conditional_var_templates)
    var_templates = conditioning_var_templates + conditional_var_templates
    gaussian_parameters = {'K': nlg.get_K(), 'h': nlg.get_h(), 'g': 0}
    return GaussianTemplate(gaussian_parameters, var_templates)


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
        self._is_vacuous = False
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
            h_error_msg = f"Error: h vector dimension inconsistency: {self.h.shape[0]} (should be {self.dim})"
            assert self.h.shape[0] == self.dim, h_error_msg
            self.g = _factor_utils.make_scalar(g)
            self.cov, self.mean, self.log_weight = None, None, None
            self.CANFORM = True
            self.COVFORM = False

            # Note: this is important to allow vacuous Gaussians (that arise, for exmaple, when Gaussians are divided by
            # identical distributions or with unconditioned nonlinear Gaussians) to be merginalised.
            if self.is_vacuous:
                self._is_vacuous = True

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

        if not np.isclose(self.get_weight(), gaussian.get_weight(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        gaussian._reorder_parameters(self.var_names)
        if not np.allclose(self.mean, gaussian.get_mean(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        if not np.allclose(self.cov, gaussian.get_cov(), rtol=rtol, atol=atol, equal_nan=False):
            return False
        return True

    # pylint: enable=protected-access

    def equals(self, factor, rtol=1e-04, atol=1e-04):
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

        # TODO: extend this to cover other factors that could be equal (i.e Gaussian mixtures with one component or
        #  Gaussian factorised factors)
        if not isinstance(factor, Gaussian):
            raise ValueError(f'unexpected factor type {type(factor)} in Gaussian equals function.')

        if set(self._var_names) != set(factor.var_names):
            return False

        gaussian_copy = factor.copy()
        # pylint: disable=protected-access
        if self._var_names != factor.var_names:
            gaussian_copy._reorder_parameters(self._var_names)
        if gaussian_copy._is_vacuous and self._is_vacuous:
            return True
        if self.CANFORM:
            if self._canform_equals(factor, rtol, atol):
                return True
            return False
        self._update_covform()
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
        :rtype: dxd numpy.ndarray or float
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

    def _cov_exists(self):
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

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        :rtype: Gaussian
        """
        gaussian_copy = self.copy()
        gaussian_copy._update_covform()
        gaussian_copy.log_weight = 0.0
        gaussian_copy._destroy_canform()
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
        gaussian_copy._destroy_covform()
        return gaussian_copy

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
        Get the weight of the distribution - the value of the integral of the (possibly unnormalized) distribution.

        :return: The weight.
        :rtype: float
        """
        return np.exp(self.get_log_weight())

    def marginalize(self, vrs, keep=False):
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
        if self._is_vacuous:
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
            inv_Kyy = np.linalg.inv(Kyy)
            hx = self.h[xx_indcs]
            hy = self.h[yy_indcs]
            Kxy_inv_Kyy = Kxy.dot(np.linalg.inv(Kyy))
            K = Kxx - Kxy_inv_Kyy.dot(Kyx)
            h = hx - Kxy_inv_Kyy.dot(hy)
            g = self.g + 0.5 * ((hy.T.dot(np.linalg.inv(Kyy)).dot(hy)) + np.log(np.linalg.det(2.0 * np.pi * inv_Kyy)))
            return Gaussian(K=K, h=h, g=g, var_names=vars_to_keep)

        assert self.COVFORM
        indices_to_keep = [self._var_names.index(variable) for variable in vars_to_keep]
        marginal_var_names = vars_to_keep.copy()

        marginal_cov = self.cov[np.ix_(indices_to_keep, indices_to_keep)]
        marginal_mean = self.mean[np.ix_(indices_to_keep, [0])]
        return Gaussian(cov=marginal_cov, mean=marginal_mean, log_weight=self.log_weight,
                        var_names=marginal_var_names)

    def _destroy_canform(self):
        self.K, self.h, self.h = None, None, None
        self.CANFORM = False

    def _destroy_covform(self):
        self.cov, self.mean, self.log_weight = None, None, None
        self.COVFORM = False

    def _update_canform(self):
        """
        Update the canonical form parameters of the Gaussian.
        """
        # pylint: disable=invalid-name
        if self.CANFORM:
            return
        assert self.COVFORM
        self.K = _factor_utils.inv_matrix(self.cov)
        self.h = self.K.dot(self.mean)
        uTKu = ((self.mean.transpose()).dot(self.K)).dot(self.mean)

        det2picov = np.linalg.det(2.0 * np.pi * self.cov)
        g_ = self.log_weight - 0.5 * uTKu - 0.5 * _factor_utils.log(det2picov)

        self.g = _factor_utils.make_scalar(g_)
        self.CANFORM = True
        # pylint: enable=invalid-name

    def _update_covform(self):
        """
        Update the covariance form parameters of the Gaussian.
        """
        if self._is_vacuous:
            raise Exception('cannot update covariance form for vacuous Gaussian.')
        # pylint: disable=invalid-name
        if self.COVFORM:
            return
        assert self.CANFORM
        self.cov = _factor_utils.inv_matrix(self.K)
        assert not np.isnan(np.sum(self.cov)), 'Error: nan values in cov matrix after inversion.'
        self.mean = self.cov.dot(self.h)
        uTKu = ((self.mean.transpose()).dot(self.K)).dot(self.mean)
        det_2_pi_cov = np.linalg.det(2.0 * np.pi * self.cov)
        log_weight_ = self.g + 0.5 * uTKu + 0.5 * np.log(det_2_pi_cov)
        self.log_weight = _factor_utils.make_scalar(log_weight_)
        self.COVFORM = True
        # pylint: enable=invalid-name

    def multiply(self, factor):
        """
        Multiply this Gaussian with another factor.

        :param Gaussian factor: the factor to multiply with
        :return: the resulting factor
        """
        # if isinstance(factor, NonLinearGaussian):
        #    return factor.multiply(self)
        return self._absorb_or_cancel(factor, operator.add)

    def divide(self, factor):
        """
        Divide this Gaussian by another factor.

        :param Gaussian factor: the factor to divide by
        :return: the resulting factor
        """
        return self._absorb_or_cancel(factor, operator.sub)

    def _absorb_or_cancel(self, factor, operator_function):
        """
        A general function which can either perform Gaussian multiplication or division (which are very similar).

        :param factor: the gaussian to multiply or divide by
        :param operator_function: the operator to use one the Gaussian canonical parameters
                        ('+' for multiplication and '-' for division)
        :return: the resulting Gaussian
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
        A helper function for reduce.

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
    def reduce(self, vrs, values):
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
        Note: here it does not matter if we take KL(P||Q) or KL(Q||P) the result is either 0.0 (if both are vacuous)
        or inf (if one is not).

        :return: The KL divergence.
        :rtype: float
        """
        if self._is_vacuous:
            return 0.0
        else:
            return float('inf')

    def kl_divergence(self, factor, normalize_factor=True):
        """
        Get the KL-divergence D_KL(self || factor) between a normalized version of this factor and another factor.
        Reference https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf, page 1.

        :param factor: The other factor
        :type factor: Gaussian
        :param normalize_factor: Whether or not to normalize the other factor before computing the KL-divergence.
        :type normalize_factor: bool
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        if self.dim != factor.dim:
            raise ValueError('cannot calculate KL-divergence between Gaussians of different dimensionalities.')
        if self._is_vacuous and factor._is_vacuous:
            return 0.0
        if self._is_vacuous or factor._is_vacuous:
            return np.inf

        if self.equals(factor):
            return 0.0
        # TODO: can we compute the correct ('normalised') KL divergence without explicitly normalizing?
        normalized_self = self.normalize()
        factor_ = factor
        if normalize_factor:
            factor_ = factor.normalize()
        inv_cov_q = factor_.get_K()
        inv_cov_p = normalized_self.get_K()
        cov_p = normalized_self.get_cov()

        u_q = factor_.get_mean()
        u_p = normalized_self.get_mean()

        det_inv_cov_q = np.linalg.det(inv_cov_q)
        det_inv_cov_p = np.linalg.det(inv_cov_p)
        assert det_inv_cov_q != 0.0, 'Unexpected factor covariance determinant of 0.'
        det_term = 0.5 * cmath.log(det_inv_cov_p / det_inv_cov_q)
        trace_term = 0.5 * np.trace(inv_cov_q.dot(cov_p))
        mahalanobis_term = 0.5 * (u_p - u_q).T.dot(inv_cov_q).dot(u_p - u_q)
        dim_term = 0.5 * normalized_self.dim
        kld = det_term + trace_term + mahalanobis_term - dim_term
        # TODO: Add warning or error if this is negative and remove abs below
        return np.abs(kld[0][0])

    @property
    def is_vacuous(self):
        """
        Check if a Gaussian distribution contains no information. This is the case when the K matrix is a zero matrix.

        :return: The result of the check.
        :rtype: Bool
        """

        if self._is_vacuous:
            return True
        if self.CANFORM:
            if np.allclose(self.K, 0.0):
                if not _factor_utils.is_pos_def(self.K):
                    return True
        return False

    def copy(self):
        """
        Make a copy of this Gaussian.

        :return: The copied Gaussian.
        :rtype: Gaussian
        """

        if self.COVFORM and self.CANFORM:
            assert isinstance(self.g, (int, float))
            assert isinstance(self.log_weight, (int, float))
            gaussian_copy = Gaussian(cov=self.cov.copy(),
                                     mean=self.mean.copy(),
                                     log_weight=self.log_weight,
                                     var_names=copy.deepcopy(self._var_names))
            gaussian_copy.K = self.K.copy()
            gaussian_copy.h = self.h.copy()
            gaussian_copy.g = self.g
            gaussian_copy.CANFORM = True
            return gaussian_copy

        if self.COVFORM:
            assert isinstance(self.log_weight, (int, float))
            return Gaussian(cov=self.cov.copy(),
                            mean=self.mean.copy(),
                            log_weight=self.log_weight,
                            var_names=copy.deepcopy(self._var_names))
        if self.CANFORM:
            assert isinstance(self.g, (int, float))
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

        :param x_val: the vector (or vector-like object) to evaluate the Gaussian at
        :type x_val: vector-like
        :param vrs: The variables corresponding to the values in x_val.
        :type vrs: str list
        :return: The log of the value of the Gaussian potential at x_val.
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
            K = _factor_utils.inv_matrix(self.cov)
            exponent = ((-0.5 * (self.mean - x_vec).transpose()).dot(K)).dot(self.mean - x_vec)
            log_potx = log_norm + exponent

        if self.CANFORM:
            log_potx = -0.5 * (x_vec.transpose().dot(self.K)).dot(x_vec) + x_vec.transpose().dot(self.h) + self.g

        return log_potx[0, 0]

    # pylint: enable=invalid-name

    def _get_cov_repr_str(self):
        self_copy = self.copy()
        self_copy._update_covform()
        np.set_printoptions(linewidth=np.inf)
        repr_str = 'Cov        = \n' + str(self_copy.cov) + '\n' + \
                   'mean       = \n' + str(self_copy.mean) + '\n' + \
                   'log_weight = \n' + str(self_copy.log_weight) + '\n'
        return repr_str

    def _get_can_repr_str(self):
        self_copy = self.copy()
        self_copy._update_canform()
        np.set_printoptions(linewidth=np.inf)
        repr_str = 'K = \n' + str(self_copy.K) + '\n' + \
                   'h = \n' + str(self_copy.h) + '\n' + \
                   'g = \n' + str(self_copy.g) + '\n' + \
                   'is_vacuous: ' + str(self_copy._is_vacuous) + '\n'
        return repr_str

    def __repr__(self):  # pragma: no cover
        """
        Get the string representation of the Gaussian factor.
        """
        np.set_printoptions(edgeitems=3)
        np.set_printoptions(precision=4)
        np.core.arrayprint._line_width = 200
        repr_str = 'vars = ' + str(self.var_names) + '\n'
        if not self._is_vacuous:
            repr_str += self._get_can_repr_str()
        repr_str += self._get_cov_repr_str()
        return repr_str

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
        if not self._is_vacuous and update_covform:
            self_copy._update_covform()
        print('vars = ', self_copy.var_names)
        if self_copy.COVFORM:
            print(self_copy._get_cov_repr_str())
        if self_copy.CANFORM and show_canform:
            print(self_copy._get_can_repr_str())

    def show_vis(self, figsize=(10, 8)):
        """
        Visualise the parameters with plots.

        :param figsize: The figure size.
        :type figsize: 2 element int tuple
        """
        # TODO: find a way of making cov matrix square.
        f, [ax_cov, ax_mean] = plt.subplots(nrows=2, figsize=figsize)
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
                lb = self.get_mean()[0, 0] - 3.0 * stddev
                ub = self.get_mean()[0, 0] + 3.0 * stddev
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

    # TODO: Generalise this to more than one dimensional cases.
    def _split_gaussian(self):
        """
        Split this factor into a three component Gaussian Mixture (with all components having different means) where the
        new mixture has the same mean and covariance as the original Gaussian.

        :return: The split Gaussian Mixture.
        :rtype: GaussianMixture
        """
        from veroku.factors.experimental.gaussian_mixture import GaussianMixture
        if self.dim != 1:
            raise NotImplementedError('Gaussian must be one dimensional.')
        weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        full_cov = self.get_cov()
        full_mean = self.get_mean()

        covs = [w * full_cov for w in weights]
        means = [full_mean - np.sqrt(full_cov), full_mean, full_mean + np.sqrt(full_cov)]

        gaussians = []
        for mean, cov, weight in zip(means, covs, weights):
            gaussians.append(Gaussian(cov=cov,
                                      mean=mean,
                                      log_weight=np.log(weight),
                                      var_names=self.var_names))
        return GaussianMixture(gaussians)


class GaussianTemplate(FactorTemplate):

    def __init__(self, gaussian_parameters, var_templates):
        """
        Create a Categorical factor template.

        :param gaussian_parameters: The cononical Gaussian parameters (see example below)
        :type gaussian_parameters: str to np.array/float dict
        :param var_templates: A list of formattable strings.
        :type var_templates: str list

        >>>gaussian_parameters = {'K': np.array([[1,0],[0,1]]),
        >>>                       'h': np.array([[0],[1]]),
        >>>                       'g': 0}
        """

        super().__init__(var_templates=var_templates)
        self.K = gaussian_parameters['K']
        self.h = gaussian_parameters['h']
        self.g = gaussian_parameters['g']

    def make_factor(self, format_dict=None, var_names=None):
        """
        Make a factor with var_templates formatted by format_dict to create specific var names.

        :param format_dict: The dictionary to be used to format the var_templates strings.
        :type format_dict: str dict
        :return: The instantiated factor.
        :rtype: Gaussian
        """
        if format_dict is not None:
            assert var_names is None
            var_names = [vt.format(**format_dict) for vt in self._var_templates]
        # TODO: remove this and find better solution
        g = Gaussian(K=self.K.copy(), h=self.h.copy(), g=self.g, var_names=var_names)
        g._is_vacuous = True
        return g
