"""
Tests for the SparseLogTable module.
"""

# System imports
import unittest

# Third-party imports
import numpy as np

# Local imports
from veroku.factors.categorical import Categorical


class TestCategorical(unittest.TestCase):
    """
    Tests for Categorical class.
    """

    def setUp(self):
        """
        Set up some variables.
        """
        vars_a = ['a', 'b', 'c']
        probs_a = {(0, 0, 0): 0.01,
                   (0, 0, 1): 0.02,
                   (0, 1, 0): 0.03,
                   (0, 1, 1): 0.04,
                   (1, 0, 0): 0.05,
                   (1, 0, 1): 0.06,
                   (1, 1, 0): 0.07,
                   (1, 1, 1): 0.08}
        self.sp_table_a = Categorical(var_names=vars_a, log_probs_table=probs_a, cardinalities=[2, 2, 2])
        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.1,
                   (0, 1): 0.2,
                   (1, 0): 0.3,
                   (1, 1): 0.4}
        self.sp_table_b = Categorical(var_names=vars_b, log_probs_table=probs_b, cardinalities=[2, 2, 2])

    def test_absorb(self):
        """
        Test that the absorb function returns the correct result.
        """
        vars_ex = ['a', 'b', 'c']
        probs_ex = {(0, 0, 0): 0.11,
                    (0, 0, 1): 0.12,
                    (0, 1, 0): 0.23,
                    (0, 1, 1): 0.24,
                    (1, 0, 0): 0.35,
                    (1, 0, 1): 0.36,
                    (1, 1, 0): 0.47,
                    (1, 1, 1): 0.48}
        expected_resulting_factor = Categorical(var_names=vars_ex, log_probs_table=probs_ex, cardinalities=[2, 2, 2])
        actual_resulting_factor = self.sp_table_a.absorb(self.sp_table_b)
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    # TODO: change to log form and fix
    def test_marginalise(self):
        """
        Test that the marginalise function returns the correct result.
        """
        vars_a = ['a', 'b', 'c']
        probs_a = {(0, 0, 0): np.log(0.01),
                   (0, 0, 1): np.log(0.02),
                   (0, 1, 0): np.log(0.03),
                   (0, 1, 1): np.log(0.04),
                   (1, 0, 0): np.log(0.05),
                   (1, 0, 1): np.log(0.06),
                   (1, 1, 0): np.log(0.07),
                   (1, 1, 1): np.log(0.08)}
        sp_table_a = Categorical(var_names=vars_a, log_probs_table=probs_a, cardinalities=[2, 2, 2])
        vars_ex = ['a']
        probs_ex = {(0,): np.log(0.10),
                    (1,): np.log(0.26)}
        expected_resulting_factor = Categorical(var_names=vars_ex, log_probs_table=probs_ex, cardinalities=[2, 2, 2])
        actual_resulting_factor = sp_table_a.marginalise(vrs=['b', 'c'])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

        vars_ex = ['c']
        probs_ex = {(0,): np.log(0.01+0.03+0.05+0.07),
                    (1,): np.log(0.02+0.04+0.06+0.08)}
        expected_resulting_factor = Categorical(var_names=vars_ex, log_probs_table=probs_ex, cardinalities=[2, 2, 2])
        actual_resulting_factor = sp_table_a.marginalise(vrs=['a', 'b'])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_observe_1(self):
        """
        Test that the observe function returns the correct result.
        """

        vars_a = ['a', 'b', 'c']
        probs_a = {(0, 0, 0): 0.01,
                   (0, 0, 1): 0.02,
                   (0, 1, 0): 0.03,
                   (0, 1, 1): 0.04,
                   (1, 0, 0): 0.05,
                   (1, 0, 1): 0.06,
                   (1, 1, 0): 0.07,
                   (1, 1, 1): 0.08}
        sp_table_a = Categorical(var_names=vars_a, log_probs_table=probs_a, cardinalities=[2, 2, 2])
        vars_ex = ['b', 'c']
        probs_ex = {(0, 0): 0.01,
                    (0, 1): 0.02,
                    (1, 0): 0.03,
                    (1, 1): 0.04}
        expected_resulting_factor = Categorical(var_names=vars_ex, log_probs_table=probs_ex, cardinalities=[2, 2, 2])
        actual_resulting_factor = sp_table_a.observe(vrs=['a'], values=[0])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_observe_2(self):
        """
        Test that the observe function returns the correct result.
        """

        vars_a = ['a', 'b', 'c']
        probs_a = {(0, 0, 0): 0.01,
                   (0, 0, 1): 0.02,
                   (0, 1, 0): 0.03,
                   (0, 1, 1): 0.04,
                   (1, 0, 0): 0.05,
                   (1, 0, 1): 0.06,
                   (1, 1, 0): 0.07,
                   (1, 1, 1): 0.08}

        sp_table_a = Categorical(var_names=vars_a, log_probs_table=probs_a, cardinalities=[2, 2, 2])
        vars_ex = ['a', 'c']
        probs_ex = {(0, 0): 0.03,
                    (0, 1): 0.04,
                    (1, 0): 0.07,
                    (1, 1): 0.08}
        expected_resulting_factor = Categorical(var_names=vars_ex, log_probs_table=probs_ex, cardinalities=[2, 2, 2])
        actual_resulting_factor = sp_table_a.observe(vrs=['b'], values=[1])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_distance_from_vacuous(self):
        """
        Test that the distance from vacuous function returns the correct result.
        """
        vars = ['a', 'c']
        probs = {(0, 1): np.log(0.4),
                 (1, 0): np.log(0.2),
                 (1, 1): np.log(0.3)}
        factor = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[2, 2])

        correct_KL_p_vac = sum([(0.4/0.9)*(np.log(0.4/0.9)-np.log(0.25)),
                                (0.2/0.9)*(np.log(0.2/0.9)-np.log(0.25)),
                                (0.3/0.9)*(np.log(0.3/0.9)-np.log(0.25))])
        calculated_KL_p_vac = factor.distance_from_vacuous()
        self.assertAlmostEqual(calculated_KL_p_vac, correct_KL_p_vac)

    def test_distance_from_vacuous_sparse(self):
        """
        Test that the distance from vacuous function returns the correct result.
        """
        vars = ['a', 'c']
        probs = {(0, 1): np.log(0.5),
                 (1, 0): np.log(0.2),
                 (1, 1): np.log(0.3)}
        factor = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[2, 2])

        correct_KL_p_vac = sum([0.5*(np.log(0.5)-np.log(0.25)),
                                0.2*(np.log(0.2)-np.log(0.25)),
                                0.3*(np.log(0.3)-np.log(0.25))])
        calculated_KL_p_vac = factor.distance_from_vacuous()
        self.assertAlmostEqual(calculated_KL_p_vac, correct_KL_p_vac)

    def test_kld(self):
        """
        Test that the kld function returns the correct result.
        """
        vars = ['a']
        probs = {(2,): np.log(0.2),
                 (4,): np.log(0.8)}
        factor_1 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): np.log(0.3),
                 (4,): np.log(0.7)}
        factor_2 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld = 0.2*(np.log(0.2) - np.log(0.3)) + 0.8*(np.log(0.8) - np.log(0.7))
        self.assertAlmostEqual(correct_kld, computed_kld)

    def test_kld2(self):
        """
        Test that the kld function returns the correct result.
        """
        vars = ['a']
        probs = {(2,): np.log(1.0)}
        factor_1 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): np.log(1.0),
                 (4,): np.log(1.0)}
        factor_2 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld = 1.0*(np.log(1.0) - np.log(0.5))

    def test_kld3(self):
        """
        Test that the kld function returns the correct result.
        """
        vars = ['a']
        probs = {(2,): np.log(1.0),
                 (4,): np.log(1e-5)}
        factor_1 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): np.log(1.0),
                 (4,): np.log(1.0)}
        factor_2 = Categorical(var_names=vars, log_probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld_1 = 1.0*(np.log(1.0) - np.log(0.5))
        correct_kld_2 = 1e-5*(np.log(1e-5) - np.log(0.5))
        correct_kld = correct_kld_1 + correct_kld_2
        self.assertAlmostEqual(computed_kld, correct_kld, places=4)






