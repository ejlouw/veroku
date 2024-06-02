
# Standard imports
import unittest
from collections import namedtuple

# Third-party imports
import numpy as np

from veroku.cluster_graph import ClusterGraph
# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors.gaussian_wishart import GaussianWishart
from veroku.factors.generalized_categorical import GeneralizedCategorical

from veroku.factors.gaussian_mixture import GaussianMixture
from veroku.factors.unknown_gaussian import UnknownGaussian


# pylint: disable=no-self-use
class TestGaussianMixtureParameterInference(unittest.TestCase):
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
        # prior
        factors_table = {}
        for i in range(3):
            var_names = [f"mu_{i}", f"K_{i}"]
            gw = GaussianWishart(
                v=3,
                inv_V=np.ones([2,2]),
                lambda_0=1,
                mu_0=np.zeros([2,1]),
                log_weight=0.0,
                var_names=var_names
                )
            factors_table[(i,)] = gw
        gwc_prior = GeneralizedCategorical(var_names=["c"], cardinalities=(3,), factors_table=factors_table)

        gt_gaussian_0 = Gaussian(mean=[0,0], cov=[[2,1],[1, 2]], log_weight=0.0, var_names=["a", "b"])
        gt_gaussian_1 = Gaussian(mean=[1,1], cov=[[2,0],[0, 2]], log_weight=0.0, var_names=["a", "b"])
        gt_gaussian_2 = Gaussian(mean=[2,5], cov=[[5,2],[2, 4]], log_weight=0.0, var_names=["a", "b"])
        gt_gm = GaussianMixture([gt_gaussian_0, gt_gaussian_1, gt_gaussian_2])
        X = gt_gm.sample(100).T

        unknown_gaussians = [UnknownGaussian([f"mu_{i}", f"K_{i}", "x"]) for i in range(3)]
        factors = [gwc_prior] + unknown_gaussians
        cg = ClusterGraph(factors)
