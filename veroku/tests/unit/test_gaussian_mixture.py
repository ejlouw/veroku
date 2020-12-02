"""
A Test module for the GaussianMixture class
"""
# System imports
import unittest

# Third-party imports
import numpy as np

# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors.gaussian_mixture import GaussianMixture


def get_random_gaussian(cov_coeff, mean_coeff=1.0, seed=None):
    """
    A test helper function that generates random Gaussian factors.
    :param cov_coeff: The scale coefficient for the uniform distribution that the variance parameter is drawn from.
    :param mean_coeff:  The scale coefficient for the uniform distribution that the mean parameter is drawn from.
    :return: a random Gaussian factor
    """
    if seed is not None:
        np.random.seed(seed)
    cov = np.random.rand() * cov_coeff
    mean = np.random.rand() * mean_coeff
    weight = np.random.rand()
    random_gaussian = Gaussian(cov=cov, mean=mean, log_weight=np.log(weight), var_names=['a'])
    return random_gaussian


def get_random_gaussian_mixture(cov_coeff=1.0, mean_coeff=1.0, num_components=3, seed=None):
    """
    A test helper function that generates random Gaussian factors.
    :param n: the number of components.
    :param cov_coeff: The scale coefficient for the uniform distribution that the variance parameter is drawn from.
    :param mean_coeff:  The scale coefficient for the uniform distribution that the mean parameter is drawn from.
    :return: a random Gaussian factor
    """
    random_gaussians = []
    for i in range(num_components):
        if seed is not None:
            comp_seed = seed*i
        random_gaussians.append(get_random_gaussian(cov_coeff, mean_coeff, seed=comp_seed))
    return GaussianMixture(random_gaussians)


class TestGaussianMixture(unittest.TestCase):
    """
    Tests for the GaussianMixture class
    """

    def setUp(self):

        self.gaussian_ab_1 = Gaussian(cov=np.eye(2), mean=[1, 1], log_weight=1.0, var_names=['a', 'b'])
        self.gaussian_ab_2 = Gaussian(cov=np.eye(2), mean=[2, 2], log_weight=2.0, var_names=['a', 'b'])
        self.gaussian_ab_3 = Gaussian(cov=np.eye(2), mean=[3, 3], log_weight=3.0, var_names=['a', 'b'])
        self.gaussian_ab_4 = Gaussian(cov=np.eye(2), mean=[3, 3], log_weight=3.0, var_names=['a', 'b'])

        self.gaussian_cd_1 = Gaussian(cov=np.eye(2), mean=[1, 1], log_weight=1.0, var_names=['c', 'd'])

        self.gaussian_mixture_ab_34 = GaussianMixture(factors=[self.gaussian_ab_3, self.gaussian_ab_4])

        self.gaussian_mixture_ab_12 = GaussianMixture(factors=[self.gaussian_ab_1, self.gaussian_ab_2])
        self.gaussian_mixture_ab_123 = GaussianMixture(factors=[self.gaussian_ab_1,
                                                                self.gaussian_ab_2,
                                                                self.gaussian_ab_3])

        self.gaussian_mixture_ab_1_copy = GaussianMixture(factors=[self.gaussian_ab_1, self.gaussian_ab_2])

    def test_invalid_construction(self):
        """
        Test that constructor fails with inconsistent scope Gaussian.
        """
        with self.assertRaises(ValueError):
            GaussianMixture([self.gaussian_ab_1, self.gaussian_cd_1])

    def test_equals_true(self):
        """
        Test that the equals function returns true for identical Gaussian mixtures.
        """
        self.assertTrue(self.gaussian_mixture_ab_12.equals(self.gaussian_mixture_ab_1_copy))

    def test_equals_false(self):
        """
        Test that the equals function returns false for different Gaussian mixtures.
        """
        self.assertFalse(self.gaussian_mixture_ab_12.equals(self.gaussian_mixture_ab_34))

    def test_absorb(self):
        """
        Test that the multiply function results in the correct components.
        """
        expected_product_components = [self.gaussian_ab_1.multiply(self.gaussian_ab_3),
                                       self.gaussian_ab_1.multiply(self.gaussian_ab_4),
                                       self.gaussian_ab_2.multiply(self.gaussian_ab_3),
                                       self.gaussian_ab_2.multiply(self.gaussian_ab_4)]
        expected_gm = GaussianMixture(expected_product_components)

        actual_gm = self.gaussian_mixture_ab_12.multiply(self.gaussian_mixture_ab_34)
        self.assertTrue(actual_gm.equals(expected_gm))

    def test_marginalise(self):
        """
        Test that the marginalize function results in the correct marginal components.
        """
        expected_marginal_components = [self.gaussian_ab_1.marginalize(vrs='a'),
                                        self.gaussian_ab_2.marginalize(vrs='a')]
        expected_gm = GaussianMixture(expected_marginal_components)
        actual_gm = self.gaussian_mixture_ab_12.marginalize(vrs='a')
        self.assertTrue(actual_gm.equals(expected_gm))

    def test_normalize(self):
        """
        Test that the normalize function results in a a mixture  with a weight of 1.0.
        :return:
        """
        normed_gm = self.gaussian_mixture_ab_123.normalize()
        log_weight = normed_gm.get_log_weight()
        actual_weight = np.exp(log_weight)
        self.assertAlmostEqual(actual_weight, 1.0)

    def test_moment_match_to_single_gaussian(self):
        """
        Test that the moment matching function returns a single Gaussian with the correct parameters.
        """
        # TODO: add better test, with different means, here.
        gaussian_ab_5 = Gaussian(cov=[[2, 1], [1, 2]], mean=[0, 0], log_weight=np.log(0.5), var_names=['a', 'b'])
        gaussian_ab_6 = Gaussian(cov=[[6, 3], [3, 6]], mean=[0, 0], log_weight=np.log(0.5), var_names=['a', 'b'])
        gaussian_mixture = GaussianMixture([gaussian_ab_5, gaussian_ab_6])
        actual_gaussian = gaussian_mixture.moment_match_to_single_gaussian()
        expected_gaussian = Gaussian(cov=[[4, 2], [2, 4]], mean=[0, 0], log_weight=np.log(1.0), var_names=['a', 'b'])
        self.assertTrue(actual_gaussian.equals(expected_gaussian))

    def test_cancel_method_1(self):
        """
        Test that the _gm_division_m2 function returns
        :return:
        """
        # TODO: improve this test.

        gaussian_mixture_1 = get_random_gaussian_mixture(cov_coeff=10, seed=1)
        gaussian_mixture_2 = get_random_gaussian_mixture(cov_coeff=10, seed=2)
        gaussian_mixture_12 = gaussian_mixture_1.multiply(gaussian_mixture_2)

        gaussian_mixture_12.cancel_method = 1
        gaussian_mixture_quotient_approximation = gaussian_mixture_12.divide(gaussian_mixture_2)

    def test_cancel_method_2(self):
        """
        Test that the _gm_division_m2 function returns
        :return:
        """
        # TODO: improve this test.

        gaussian_mixture_1 = get_random_gaussian_mixture(cov_coeff=10, seed=1)
        gaussian_mixture_2 = get_random_gaussian_mixture(cov_coeff=10, seed=2)
        gaussian_mixture_12 = gaussian_mixture_1.multiply(gaussian_mixture_2)

        gaussian_mixture_12.cancel_method = 2
        gaussian_mixture_quotient_approximation = gaussian_mixture_12.divide(gaussian_mixture_2)
        self.assertEqual(gaussian_mixture_quotient_approximation.num_components, 9)
