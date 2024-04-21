
# Standard imports
import unittest
from collections import namedtuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.integrate import dblquad

# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors.gaussian_wishart import GaussianWishart
from veroku.factors.generalized_categorical import GeneralizedCategorical
from veroku.factors.generlized_mixture import GeneralizedMixture
from veroku.factors.sparse_categorical import SparseCategorical


# pylint: disable=no-self-use
class TestGeneralisedMixture(unittest.TestCase):
    """
    Tests for the GaussianMixture class
    """

    def setUp(self):
        """
        Run before every test.
        """
        pass

    def test_1D_multiplication(self):
        """
        Test that multiplication of 1D Gaussian-Wishart factors returns the correct result.
        """
        test_data_df = pd.read_csv("./tests/test_data/gaussian_wishart_1d_mulitplication.csv")
        for i, example_df in test_data_df.groupby('example_index'):
            factors_params_dict = {}
            for factor_label, factor_df in example_df.groupby("factor_label"):
                factor_param_df = factor_df.drop(columns=["factor_label", "example_index"])
                factor_param_dict = list(factor_param_df.T.to_dict().values())[0]
                factors_params_dict[factor_label] = factor_param_dict

            factor_a_params = factors_params_dict["factor_a"]
            factor_a_params["var_names"] = eval(factor_a_params["var_names"])
            factor_a = GaussianWishart(**factor_a_params)

            factor_b_params = factors_params_dict["factor_b"]
            factor_b_params["var_names"] = eval(factor_b_params["var_names"])
            factor_b = GaussianWishart(**factor_b_params)

            expected_product_params = factors_params_dict["ab_product"]
            expected_product_params["var_names"] = eval(expected_product_params["var_names"])
            expected_product_factor = GaussianWishart(**expected_product_params)

            actual_product_factor = factor_a.absorb(factor_b)
            assert actual_product_factor == expected_product_factor
            
    #def test_normalized(self):
    #    """
    #    Test that a normalized GaussianWishart factor integrates to 1.0.
    #    """
    #    np.random.seed(0)
    #    x = np.random.normal(0, 10, (100, 2))
    #    inv_V = np.linalg.inv(np.cov(x.T))
    #    v = np.random.uniform(1,10)
    #    lambda_0 = np.random.uniform(1, 10)
    #    mu_0 = np.random.normal(0, 10, (1, 2))
    #    gw = GaussianWishart(v=v, inv_V=inv_V, lambda_0=lambda_0, mu_0=mu_0, log_weight=0.0, var_names=["mu", "K"])
    #
    #    result, error = dblquad(gw.potential, 1e-20, 10, lambda x2: -100, lambda x2: 100)
    #    return result


