# System imports
import unittest
import numpy as np

# Local imports
from veroku.factors.experimental.factorised_factor import FactorizedFactor
from veroku.factors.gaussian import Gaussian

"""
Test module for the factor_utils module.
"""


# pylint: disable=too-many-public-methods
class TestFactorisedFactor(unittest.TestCase):
    """
    A class for testing the FactorisedFactor class.
    """

    def setUp(self):
        self.gab = Gaussian(cov=[[3, 1], [1, 2]], mean=[1, 2], log_weight=0.0, var_names=["a", "b"])
        self.gcd = Gaussian(cov=[[3, 2], [2, 4]], mean=[3, 4], log_weight=0.0, var_names=["c", "d"])
        self.gef = Gaussian(cov=[[8, 3], [3, 6]], mean=[5, 6], log_weight=0.0, var_names=["e", "f"])

        self.gab2 = Gaussian(cov=[[4, 1], [1, 2]], mean=[1, 2], log_weight=0.0, var_names=["a", "b"])
        self.gbc2 = Gaussian(cov=[[8, 1], [1, 8]], mean=[7, 10], log_weight=0.0, var_names=["b", "c"])

        self.gab3 = Gaussian(K=[[5, 1], [1, 2]], h=[1, 2], g=0.0, var_names=["a", "b"])
        self.ga3 = Gaussian(K=[[1]], h=[1], g=0.0, var_names=["a"])

    def test_multiply_independent(self):
        """
        Test that the joint distribution has been correctly calculated.
        """
        expected_cov = [
            [3, 1, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0],
            [0, 0, 3, 2, 0, 0],
            [0, 0, 2, 4, 0, 0],
            [0, 0, 0, 0, 8, 3],
            [0, 0, 0, 0, 3, 6],
        ]
        expected_mean = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        expected_joint = Gaussian(
            cov=expected_cov, mean=expected_mean, log_weight=0.0, var_names=["a", "b", "c", "d", "e", "f"]
        )
        actual_joint_ff = FactorizedFactor([self.gab])
        actual_joint_ff = actual_joint_ff.multiply(self.gcd)
        actual_joint_ff = actual_joint_ff.multiply(self.gef)
        actual_joint_distribution = actual_joint_ff.joint_distribution
        self.assertEqual(len(actual_joint_ff.factors), 3)
        self.assertTrue(actual_joint_distribution.equals(expected_joint))

    def test_multiply_dependent(self):
        """
        Test that the joint distribution has been correctly calculated.
        """
        expected_cov = [[3.9, 0.8, 0.1], [0.8, 1.6, 0.2], [0.1, 0.2, 7.9]]
        expected_mean = [[1.5], [3.0], [9.5]]
        expected_log_weight = -3.32023108
        expected_joint = Gaussian(
            cov=expected_cov, mean=expected_mean, log_weight=expected_log_weight, var_names=["a", "b", "c"]
        )
        actual_joint_ff = FactorizedFactor([self.gab2])
        actual_joint_ff = actual_joint_ff.multiply(self.gbc2)
        self.assertEqual(len(actual_joint_ff.factors), 2)
        actual_joint_distribution = actual_joint_ff.joint_distribution
        self.assertTrue(actual_joint_distribution.equals(expected_joint))

    def test_multiply_same_scope(self):
        """
        Test that the joint distribution has been correctly calculated.
        """

        expected_K = [[0.68571429, -0.34285714], [-0.34285714, 1.17142857]]
        expected_h = [[0.0], [2.0]]
        expected_g = -7.453428163563398

        expected_joint = Gaussian(K=expected_K, h=expected_h, g=expected_g, var_names=["a", "b"])
        actual_joint_ff = FactorizedFactor([self.gab])
        actual_joint_ff = actual_joint_ff.multiply(self.gab2)
        self.assertEqual(len(actual_joint_ff.factors), 1)
        actual_joint_distribution = actual_joint_ff.joint_distribution
        self.assertTrue(actual_joint_distribution.equals(expected_joint))

    def test_multiply_subset_scope(self):
        """
        Test that the joint distribution has been correctly calculated.
        """
        expected_K = [[6.0, 1.0], [1.0, 2.0]]
        expected_h = [[2.0], [2.0]]
        expected_g = 0.0

        expected_joint = Gaussian(K=expected_K, h=expected_h, g=expected_g, var_names=["a", "b"])
        actual_joint_ff = FactorizedFactor([self.gab3])
        actual_joint_ff = actual_joint_ff.multiply(self.ga3)
        self.assertEqual(len(actual_joint_ff.factors), 1)
        actual_joint_distribution = actual_joint_ff.joint_distribution
        self.assertTrue(actual_joint_distribution.equals(expected_joint))

    def test_cancel_absorbed(self):
        """
        Test that the joint distribution has been correctly calculated.
        """
        actual_joint_ff = FactorizedFactor([self.gab])
        actual_joint_ff = actual_joint_ff.multiply(self.gcd)
        actual_joint_ff = actual_joint_ff.divide(self.gcd)
        self.assertEqual(len(actual_joint_ff.factors), 1)
        actual_joint_distribution = actual_joint_ff.joint_distribution
        self.assertTrue(actual_joint_distribution.equals(self.gab))

    def test_marginalise(self):
        """
        Test that the factorised factor is correctly marginalised.
        """
        g1_vars = ["a", "b"]
        g2_vars = ["c", "d"]
        g1_mean = np.zeros([2, 1])
        g1_cov = np.eye(2)
        g2_log_weight = 0.1
        g1 = Gaussian(mean=g1_mean, cov=g1_cov, log_weight=0.0, var_names=g1_vars)
        g2 = Gaussian(mean=np.zeros([2, 1]), cov=np.eye(2), log_weight=g2_log_weight, var_names=g2_vars)
        ff_joint = FactorizedFactor([g1, g2])
        actual_marginal = ff_joint.marginalize(g1_vars, keep=True)
        expected_marginal_comp = Gaussian(
            mean=g1_mean, cov=g1_cov, log_weight=g2_log_weight, var_names=g1_vars
        )
        expected_marginal = FactorizedFactor([expected_marginal_comp])
        self.assertTrue(expected_marginal.equals(actual_marginal))

    def test_observe(self):
        """
        Test that the observe function returns the correct result.
        """
        var_names = ["a", "b"]
        cov = np.array([[5.0, 0.0], [0.0, 4.0]])
        mean = np.array([[1.0], [2.0]])
        g1 = Gaussian(mean=mean, cov=cov, log_weight=0.0, var_names=var_names)

        var_names = ["b", "c"]
        cov = np.array([[1.0, 0.0], [0.0, 2.0]])
        mean = np.array([[3.0], [5.0]])
        g2 = Gaussian(mean=mean, cov=cov, log_weight=0.0, var_names=var_names)

        var_names = ["d", "e"]
        cov = np.array([[3.0, 0.0], [0.0, 4.0]])
        mean = np.array([[0.0], [3.0]])
        g3 = Gaussian(mean=mean, cov=cov, log_weight=0.0, var_names=var_names)

        # Reduction 1: reduce all vars of an independent factor
        observed_values_1 = [1.0, 2.0]
        observed_vars_1 = ["d", "e"]

        expected_g1 = g1.copy()
        expected_g1.log_weight = -3.3719970579700123
        expected_g2 = g2.copy()
        expected_reduced_ff_1 = FactorizedFactor([expected_g1, expected_g2])

        ff = FactorizedFactor([g1, g2, g3])
        actual_reduced_ff_1 = ff.observe(vrs=observed_vars_1, values=observed_values_1)
        self.assertTrue(expected_reduced_ff_1.equals(actual_reduced_ff_1))

        # Reduction 2
        observed_values_1 = [1.0, 2.0]
        observed_vars_1 = ["b", "e"]

        g2_reduced_vars = ["a", "c", "d"]
        g2_reduced_cov = np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        g2_reduced_mean = np.array([[1.0], [5.0], [0.0]])
        g2_reduced_log_weight = -6.39311
        expected_g2_reduced_comp = Gaussian(
            cov=g2_reduced_cov,
            mean=g2_reduced_mean,
            log_weight=g2_reduced_log_weight,
            var_names=g2_reduced_vars,
        )
        expected_reduced_ff_2 = FactorizedFactor([expected_g2_reduced_comp])
        actual_reduced_ff_2 = ff.reduce(vrs=observed_vars_1, values=observed_values_1)
        self.assertTrue(expected_reduced_ff_2.equals(actual_reduced_ff_2))
