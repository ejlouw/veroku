
# Standard imports
import unittest
from collections import namedtuple

# Third-party imports
import numpy as np

# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors.generalized_categorical import GeneralizedCategorical
from veroku.factors.generlized_mixture import GeneralizedMixture
from veroku.factors.sparse_categorical import SparseCategorical


# pylint: disable=no-self-use
class TestGeneralisedCategorical(unittest.TestCase):
    """
    Tests for the GeneralisedCategorical class
    """

    def setUp(self):
        """
        Run before every test.
        """
        self.gaussian_0 = Gaussian(
            cov=np.eye(2), mean=[1, 1], log_weight=np.log(0.5), var_names=["a", "b"]
        )
        self.gaussian_1 = Gaussian(
            cov=np.eye(2), mean=[2, 2], log_weight=np.log(0.2), var_names=["a", "b"]
        )
        self.gaussian_2 = Gaussian(
            cov=np.eye(2), mean=[3, 3], log_weight=np.log(0.3), var_names=["a", "b"]
        )

        gaussian_1_pow_2 = self.gaussian_0.multiply(self.gaussian_0)
        gaussian_2_pow_2 = self.gaussian_1.multiply(self.gaussian_1)
        gaussian_3_pow_2 = self.gaussian_2.multiply(self.gaussian_2)

        factors_table = {
            (0,):self.gaussian_0,
            (1,):self.gaussian_1,
            (2,): self.gaussian_2
        }
        self.generalised_categorical = GeneralizedCategorical(var_names=['c'],
                                                              cardinalities=[3],
                                                              factors_table=factors_table)
        self.expected_mixture_marginal = GeneralizedMixture(
            [self.gaussian_0, self.gaussian_1, self.gaussian_2]
        )

        self.generalised_categorical = GeneralizedCategorical(var_names=['c'],
                                                              cardinalities=[3],
                                                              factors_table=factors_table)
        squared_factors_table = {
            (0,): gaussian_1_pow_2,
            (1,): gaussian_2_pow_2,
            (2,): gaussian_3_pow_2,
        }
        self.expected_product_factor = GeneralizedCategorical(var_names=['c'],
                                                              cardinalities=[3],
                                                              factors_table=squared_factors_table)

        discrete_marginal_log_probs_table = {
            (0,): np.log(0.5),
            (1,): np.log(0.2),
            (2,): np.log(0.3),
        }
        self.expected_discrete_marginal = SparseCategorical(
            var_names=['c'],
            cardinalities=[3],
            log_probs_table=discrete_marginal_log_probs_table)

    def test_categorical_marginalisation(self):
        """
        Test marginalising out all categorical variables returns the correct result.
        """
        mixture_marginal = self.generalised_categorical.marginalize(vrs=["c"], keep=False)
        assert self.expected_mixture_marginal.equals(mixture_marginal)

    def test_continuous_marginalisation(self):
        """
        Test marginalising out all continuous variables returns the correct result.
        """
        discrete_marginal = self.generalised_categorical.marginalize(vrs=["c"], keep=True)
        assert self.expected_discrete_marginal.equals(discrete_marginal)


    def test_categorical_multiplication(self):
        """
        Test that constructor fails with inconsistent scope Gaussian.
        """
        product_factor = self.generalised_categorical.multiply(self.generalised_categorical)
        assert product_factor.equals(self.expected_product_factor)

