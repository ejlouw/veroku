
# Standard imports
import unittest
from collections import namedtuple

# Third-party imports
import numpy as np

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

    def test_0(self):
        """
        Test marginalising out all categorical variables returns the correct result.
        """
        factors_table = {}
        for i in range(3):
            gw = GaussianWishart(v=3, inv_V=0.1, lambda_0=3, mu_0=np.random.uniform(0, 1),
                                 log_weight=0.0, var_names=["mu", "K"])
            factors_table[(i,)] = gw
        gc = GeneralizedCategorical(var_names=["a"], cardinalities=(3,),
                                    factors_table=factors_table)
        a = gc.marginalize(vrs=["a"], keep=False)
        b = gc.marginalize(vrs=["a"], keep=False)
        a.multiply(b)
