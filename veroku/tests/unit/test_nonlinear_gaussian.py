# System imports
import unittest

# Third-party imports
import numpy as np

# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors.experimental.nonlinear_gaussian import NonLinearGaussian

"""
Tests for the NonLinearGaussian module.
"""


# pylint: disable=too-many-public-methods
class TestNonLinearGaussian(unittest.TestCase):
    """
    Tests for NonlinearGaussian Class.
    """

    def test_multiply(self):
        """
        Test that the multiply function results in the correct joint distribution when the absorbed factor has the same
        scope as the conditioning scope.
        """
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=["a", "b"],
            conditional_vars=["c", "d"],
            transform=transform,
            noise_cov=noise_cov,
        )

        conditioning_cov = np.array([[3, 1], [1, 5]])
        conditioning_mean = np.array([[2], [3]])
        conditioning_gaussian = Gaussian(
            cov=conditioning_cov, mean=conditioning_mean, log_weight=0.0, var_names=["a", "b"]
        )

        # expected parameters
        S_xx = conditioning_cov
        S_xy = conditioning_cov.dot(A.T)
        S_yx = A.dot(conditioning_cov.T)
        S_yy = A.dot(conditioning_cov).dot(A.T) + noise_cov
        expected_joint_cov = np.block([[S_xx, S_xy], [S_yx, S_yy]])
        u_x = conditioning_mean
        u_y = A.dot(conditioning_mean)
        expected_joint_mean = np.block([[u_x], [u_y]])

        expected_joint = Gaussian(
            cov=expected_joint_cov, mean=expected_joint_mean, log_weight=0.0, var_names=["a", "b", "c", "d"]
        )

        nlg_factor = nlg_factor.multiply(conditioning_gaussian)
        actual_joint = nlg_factor.joint_distribution
        self.assertTrue(expected_joint.equals(actual_joint))

    def test_multiply_both_sides(self):
        """
        Test that the multiply function results in the correct joint distribution when a factor with the conditional scope
        is received first and a conditioning scope factor is received afterwards.
        """

        conditional_update_cov = np.array([[2, 1], [1, 4]])
        conditional_update_mean = np.array([[5], [4]])

        conditional_update_factor = Gaussian(
            cov=conditional_update_cov, mean=conditional_update_mean, log_weight=0.0, var_names=["c", "d"]
        )
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=["a", "b"],
            conditional_vars=["c", "d"],
            transform=transform,
            noise_cov=noise_cov,
        )

        conditioning_cov = np.array([[3, 1], [1, 5]])
        conditioning_mean = np.array([[2], [3]])
        conditioning_gaussian = Gaussian(
            cov=conditioning_cov, mean=conditioning_mean, log_weight=0.0, var_names=["a", "b"]
        )
        S_xx = conditioning_cov
        S_xy = conditioning_cov.dot(A.T)
        S_yx = A.dot(conditioning_cov.T)
        S_yy = A.dot(conditioning_cov).dot(A.T) + noise_cov
        joint_cov = np.block([[S_xx, S_xy], [S_yx, S_yy]])
        np.linalg.inv(joint_cov),

        u_x = conditioning_mean
        u_y = A.dot(conditioning_mean)
        joint_mean = np.block([[u_x], [u_y]])

        expected_joint = Gaussian(
            cov=joint_cov, mean=joint_mean, log_weight=0.0, var_names=["a", "b", "c", "d"]
        )
        expected_joint = expected_joint.multiply(conditional_update_factor.copy())

        nlg_factor = nlg_factor.multiply(conditional_update_factor)
        nlg_factor = nlg_factor.multiply(conditioning_gaussian)
        # expected_joint._update_covform()
        conditional_update_factor.show()

        print("\n\n expected_joint ")
        expected_joint.show()
        nlg_factor._recompute_joint()
        print("\n\n\nactual factor")
        nlg_factor.show()

        self.assertTrue(expected_joint.equals(nlg_factor.joint_distribution))

    def test_marginalise_unconditioned(self):
        """
        Test that marginalising an unconditioned non-linear Gaussian results in a vacuous Gaussian.
        """
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=["a", "b"],
            conditional_vars=["c", "d"],
            transform=transform,
            noise_cov=noise_cov,
        )
        actual_ab_marginal = nlg_factor.marginalize(vrs=["a", "b"], keep=True)
        expected_ab_marginal = Gaussian.make_vacuous(var_names=["a", "b"])
        self.assertTrue(actual_ab_marginal.equals(expected_ab_marginal))

    def test_subset_marginalise_unconditioned(self):
        """
        Test that marginalising an unconditioned non-linear Gaussian results in a vacuous Gaussian.
        """
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=["a", "b"],
            conditional_vars=["c", "d"],
            transform=transform,
            noise_cov=noise_cov,
        )
        actual_a_marginal = nlg_factor.marginalize(vrs=["a"], keep=True)
        expected_a_marginal = Gaussian.make_vacuous(var_names=["a"])
        self.assertTrue(actual_a_marginal.equals(expected_a_marginal))

    def test_subset2_marginalise_unconditioned(self):
        """
        Test that marginalising an unconditioned non-linear Gaussian results in a vacuous Gaussian.
        """
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=["a", "b"],
            conditional_vars=["c", "d"],
            transform=transform,
            noise_cov=noise_cov,
        )
        actual_a_marginal = nlg_factor.marginalize(vrs=["c"], keep=True)
        expected_a_marginal = Gaussian.make_vacuous(var_names=["c"])
        self.assertTrue(actual_a_marginal.equals(expected_a_marginal))

    def test_marginalise_conditioned(self):
        """
        Test that marginalising an unconditioned non-linear Gaussian results in a vacuous Gaussian.
        """
        ab_vars = ["a", "b"]
        A = np.array([[2, 0], [0, 1]])
        transform = lambda x, varnames: A.dot(x)
        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=ab_vars, conditional_vars=["c", "d"], transform=transform, noise_cov=noise_cov
        )

        cov = np.array([[2, 1], [1, 3]])
        mean = np.array([[2], [1]])
        conditioning_gaussian = Gaussian(cov=cov, mean=mean, log_weight=0.0, var_names=ab_vars)

        nlg_factor = nlg_factor.multiply(conditioning_gaussian)

        actual_ab_marginal = nlg_factor.marginalize(vrs=ab_vars, keep=True)
        expected_ab_marginal = conditioning_gaussian.copy()
        self.assertTrue(actual_ab_marginal.equals(expected_ab_marginal))

    def test_marginalise_with_conditional(self):
        """
        Test that marginalising an unconditioned non-linear Gaussian results in a vacuous Gaussian.
        """
        ab_vars = ["a", "b"]
        cd_vars = ["c", "d"]
        A = np.array([[2, 0], [0, 1]])

        def transform(x, _):
            return A.dot(x)

        noise_cov = np.array([[1, 0], [0, 1]])
        nlg_factor = NonLinearGaussian(
            conditioning_vars=ab_vars, conditional_vars=cd_vars, transform=transform, noise_cov=noise_cov
        )
        cov = np.array([[2, 1], [1, 3]])
        mean = np.array([[2], [1]])
        conditional_gaussian = Gaussian(cov=cov, mean=mean, log_weight=0.0, var_names=cd_vars)

        nlg_factor = nlg_factor.multiply(conditional_gaussian)

        actual_cd_marginal = nlg_factor.marginalize(vrs=cd_vars, keep=True)
        expected_cd_marginal = conditional_gaussian.copy()
        self.assertTrue(actual_cd_marginal.equals(expected_cd_marginal))
