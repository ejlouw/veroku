"""
Tests for the SparseLogTable module.
"""

# System imports
import unittest

# Third-party imports
import numpy as np

# Local imports
from veroku.factors.categorical import Categorical
from veroku.factors.sparse_categorical import SparseCategorical

#  TODO: add tests for strange KLD (i.e with divide by zeros)
#  TODO: add tests divide operation


def make_abc_factor_1(CatClass):
    vars_a = ['a', 'b', 'c']
    probs_a = {(0, 0, 0): np.exp(0.01),
               (0, 0, 1): np.exp(0.02),
               (0, 1, 0): np.exp(0.03),
               (0, 1, 1): np.exp(0.04),
               (1, 0, 0): np.exp(0.05),
               (1, 0, 1): np.exp(0.06),
               (1, 1, 0): np.exp(0.07),
               (1, 1, 1): np.exp(0.08)}
    return CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2, 2])


class TestCategorical(unittest.TestCase):
    """
    Tests for Categorical class.
    """

    def __init__(self, *args, **kwargs):
        """
        Set up some variables.
        """
        super().__init__(*args, **kwargs)
        self.CatClass = Categorical

    def test_absorb_same_scope(self):
        """
        Test that the multiply function returns the correct result.
        """
        vars_ex = ['a', 'b']
        probs_ex = {(0, 0): 0.01,
                    (0, 1): 0.04,
                    (1, 0): 0.09,
                    (1, 1): 0.16}
        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.1,
                   (0, 1): 0.2,
                   (1, 0): 0.3,
                   (1, 1): 0.4}
        sp_table_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])

        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2, 2])
        actual_resulting_factor = sp_table_b.multiply(sp_table_b)
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_absorb_subset_scope(self):
        """
        Test that the multiply function returns the correct result.
        """
        vars_ex = ['a', 'b', 'c']
        probs_ex = {(0, 0, 0): np.exp(0.11),
                    (0, 0, 1): np.exp(0.12),
                    (0, 1, 0): np.exp(0.23),
                    (0, 1, 1): np.exp(0.24),
                    (1, 0, 0): np.exp(0.35),
                    (1, 0, 1): np.exp(0.36),
                    (1, 1, 0): np.exp(0.47),
                    (1, 1, 1): np.exp(0.48)}
        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2, 2, 2])

        sp_table_abc = make_abc_factor_1(CatClass=self.CatClass)

        vars_b = ['a', 'b']
        probs_b = {(0, 0): np.exp(0.1),
                   (0, 1): np.exp(0.2),
                   (1, 0): np.exp(0.3),
                   (1, 1): np.exp(0.4)}
        sp_table_ab = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])
        actual_resulting_factor = sp_table_abc.multiply(sp_table_ab)
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_absorb_different_scope(self):
        """
        Test that the multiply function returns the correct result.
        """
        vars_ex = ['c', 'd', 'a', 'b']
        probs_ex = {(0, 0, 0, 0): 0.1*0.01,
                    (0, 0, 0, 1): 0.1*0.04,
                    (0, 0, 1, 0): 0.1*0.09,
                    (0, 0, 1, 1): 0.1*0.16,
                    (0, 1, 0, 0): 0.2*0.01,
                    (0, 1, 0, 1): 0.2*0.04,
                    (0, 1, 1, 0): 0.2*0.09,
                    (0, 1, 1, 1): 0.2*0.16,
                    (1, 0, 0, 0): 0.3*0.01,
                    (1, 0, 0, 1): 0.3*0.04,
                    (1, 0, 1, 0): 0.3*0.09,
                    (1, 0, 1, 1): 0.3*0.16,
                    (1, 1, 0, 0): 0.4*0.01,
                    (1, 1, 0, 1): 0.4*0.04,
                    (1, 1, 1, 0): 0.4*0.09,
                    (1, 1, 1, 1): 0.4*0.16}

        vars_ab = ['a', 'b']
        probs_ab = {(0, 0): 0.01,
                    (0, 1): 0.04,
                    (1, 0): 0.09,
                    (1, 1): 0.16}
        vars_cd = ['c', 'd']
        probs_cd = {(0, 0): 0.1,
                    (0, 1): 0.2,
                    (1, 0): 0.3,
                    (1, 1): 0.4}

        sp_table_cd = self.CatClass(var_names=vars_cd, probs_table=probs_cd, cardinalities=[2, 2])

        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2, 2, 2, 2])
        sp_table_ab = self.CatClass(var_names=vars_ab, probs_table=probs_ab, cardinalities=[2, 2])
        actual_resulting_factor = sp_table_cd.multiply(sp_table_ab)
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_cancel_different_scope(self):
        vars_ab = ['a', 'b']
        probs_ab = {(0, 0): 1.0,
                    (0, 1): 2.0,
                    (1, 0): 3.0,
                    (1, 1): 4.0}
        factor_ab = Categorical(var_names=vars_ab, probs_table=probs_ab, cardinalities=[2, 2])

        vars_b = ['b']
        probs_b = {(0,): 5.0,
                   (1,): 6.0}
        factor_b = Categorical(var_names=vars_b, probs_table=probs_b, cardinalities=[2])

        probs_ab_expected = {(0, 0): 1.0/5.0,
                             (0, 1): 2.0/6.0,
                             (1, 0): 3.0/5.0,
                             (1, 1): 4.0/6.0}
        factor_ab_expected = Categorical(var_names=vars_ab, probs_table=probs_ab_expected, cardinalities=[2, 2])

        factor_ab_actual = factor_ab.cancel(factor_b)
        self.assertTrue(factor_ab_expected.equals(factor_ab_actual))

    def test_cancel_same_scope(self):
        """
        Test the cancel function applied to factors with the same scope.
        """
        vars_a = ['a', 'b']
        probs_a = {(0, 0): 1.0,
                   (0, 1): 2.0,
                   (1, 0): 3.0,
                   (1, 1): 4.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 0): 5.0,
                   (0, 1): 6.0,
                   (1, 0): 7.0,
                   (1, 1): 8.0}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])

        vars_c = ['a', 'b']
        probs_c = {(0, 0): 1.0/5.0,
                   (0, 1): 2.0/6.0,
                   (1, 0): 3.0/7.0,
                   (1, 1): 4.0/8.0}
        expected_resulting_factor = self.CatClass(var_names=vars_c, probs_table=probs_c, cardinalities=[2, 2])
        actual_resulting_factor = factor_a.cancel(factor_b)
        actual_resulting_factor.show()
        expected_resulting_factor.show()
        self.assertTrue(expected_resulting_factor.equals(actual_resulting_factor))

    def test_cancel_with_zeros(self):
        vars_a = ['a', 'b']
        probs_a = {(0, 0): 0.0,
                   (0, 1): 0.0,
                   (1, 0): 1.0,
                   (1, 1): 1.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.0,
                   (0, 1): 1.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])

        vars_c = ['a', 'b']
        probs_c = {(0, 0): 0.0,
                   (0, 1): 0.0,
                   (1, 0): np.inf,
                   (1, 1): 1.0}
        expected_resulting_factor = self.CatClass(var_names=vars_c, probs_table=probs_c, cardinalities=[2, 2])
        actual_resulting_factor = factor_a.cancel(factor_b)
        actual_resulting_factor.show()
        expected_resulting_factor.show()
        self.assertTrue(expected_resulting_factor.equals(actual_resulting_factor))

    # TODO: change to log form and fix
    def test_marginalise(self):
        """
        Test that the marginalize function returns the correct result.
        """
        vars_a = ['a','b', 'c']
        probs_a = {(0, 0, 0): 0.01,
                   (0, 0, 1): 0.02,
                   (0, 1, 0): 0.03,
                   (0, 1, 1): 0.04,
                   (1, 0, 0): 0.05,
                   (1, 0, 1): 0.06,
                   (1, 1, 0): 0.07,
                   (1, 1, 1): 0.08}
        sp_table_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2, 2])
        vars_ex = ['a']
        probs_ex = {(0,): 0.10,
                    (1,): 0.26}
        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2])
        actual_resulting_factor = sp_table_a.marginalize(vrs=['b', 'c'])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

        vars_ex = ['c']
        probs_ex = {(0,): 0.01+0.03+0.05+0.07,
                    (1,): 0.02+0.04+0.06+0.08}
        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2])
        actual_resulting_factor = sp_table_a.marginalize(vrs=['a', 'b'])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_observe_1(self):
        """
        Test that the reduce function returns the correct result.
        """

        sp_table_abc = make_abc_factor_1(CatClass=self.CatClass)
        vars_ex = ['b', 'c']
        probs_ex = {(0, 0): np.exp(0.01),
                    (0, 1): np.exp(0.02),
                    (1, 0): np.exp(0.03),
                    (1, 1): np.exp(0.04)}
        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2, 2])
        actual_resulting_factor = sp_table_abc.reduce(vrs=['a'], values=[0])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_observe_2(self):
        """
        Test that the reduce function returns the correct result.
        """
        sp_table_abc = make_abc_factor_1(CatClass=self.CatClass )
        vars_ex = ['a', 'c']
        probs_ex = {(0, 0): np.exp(0.03),
                    (0, 1): np.exp(0.04),
                    (1, 0): np.exp(0.07),
                    (1, 1): np.exp(0.08)}
        expected_resulting_factor = self.CatClass(var_names=vars_ex, probs_table=probs_ex, cardinalities=[2, 2])
        actual_resulting_factor = sp_table_abc.reduce(vrs=['b'], values=[1])
        self.assertTrue(actual_resulting_factor.equals(expected_resulting_factor))

    def test_distance_from_vacuous(self):
        """
        Test that the distance from vacuous function returns the correct result.
        """
        vars = ['a', 'c']
        probs = {(0, 1): 0.4,
                 (1, 0): 0.2,
                 (1, 1): 0.3}
        factor = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[2, 2])

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
        probs = {(0, 1): 0.5,
                 (1, 0): 0.2,
                 (1, 1): 0.3}
        factor = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[2, 2])

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
        probs = {(2,): 0.2,
                 (3,): 0.8}
        factor_1 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): 0.3,
                 (3,): 0.7}
        factor_2 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld = 0.2*(np.log(0.2) - np.log(0.3)) + 0.8*(np.log(0.8) - np.log(0.7))
        self.assertAlmostEqual(correct_kld, computed_kld)

    def test_kld2(self):
        """
        Test that the kld function returns the correct result.
        """
        vars = ['a']
        probs = {(2,): 1.0}
        factor_1 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): 0.5,
                 (3,): 0.5}
        factor_2 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld = 1.0*(np.log(1.0) - np.log(0.5))
        self.assertAlmostEqual(computed_kld, correct_kld, places=4)

    def test_kld3(self):
        """
        Test that the kld function returns the correct result.
        """
        vars = ['a']
        probs = {(2,): 1.0,
                 (3,): 1e-5}
        factor_1 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])

        vars = ['a']
        probs = {(2,): 1.0,
                 (3,): 1.0}
        factor_2 = self.CatClass(var_names=vars, probs_table=probs, cardinalities=[4])
        computed_kld = factor_1.kl_divergence(factor_2)
        correct_kld_1 = 1.0*(np.log(1.0) - np.log(0.5))
        correct_kld_2 = 1e-5*(np.log(1e-5) - np.log(0.5))
        correct_kld = correct_kld_1 + correct_kld_2
        self.assertAlmostEqual(computed_kld, correct_kld, places=4)

    def test_KLD_with_zeros(self):
        vars_a = ['a', 'b']
        probs_a = {(0, 0): 0.0,
                   (0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.0,
                   (0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])


        expected_KLD = 0.0
        actual_KLD = factor_b.kl_divergence(factor_a)
        self.assertEqual(expected_KLD, actual_KLD)

    def test_KLD_with_zeros_sparse1(self):
        vars_a = ['a', 'b']
        probs_a = {(0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.0,
                   (0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])


        expected_KLD = 0.0
        actual_KLD = factor_b.kl_divergence(factor_a)
        self.assertEqual(expected_KLD, actual_KLD)

    def test_KLD_with_zeros_sparse2(self):
        vars_a = ['a', 'b']
        probs_a = {(0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])
        expected_KLD = 0.0
        actual_KLD = factor_b.kl_divergence(factor_a)
        self.assertEqual(expected_KLD, actual_KLD)

    def test_KLD_with_zeros_sparse3(self):
        vars_a = ['a', 'b']
        probs_a = {(0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 1.0}
        factor_a = self.CatClass(var_names=vars_a, probs_table=probs_a, cardinalities=[2, 2])

        vars_b = ['a', 'b']
        probs_b = {(0, 0): 0.5,
                   (0, 1): 0.0,
                   (1, 0): 0.0,
                   (1, 1): 0.5}
        factor_b = self.CatClass(var_names=vars_b, probs_table=probs_b, cardinalities=[2, 2])
        expected_KLD = np.log(2)
        actual_KLD = factor_a.kl_divergence(factor_b)
        self.assertEqual(expected_KLD, actual_KLD)

        expected_KLD = np.inf
        actual_KLD = factor_b.kl_divergence(factor_a)
        self.assertEqual(expected_KLD, actual_KLD)


class TestSparseCategorical(TestCategorical):
    def __init__(self, *args, **kwargs):
        """
        Set up some variables.
        """
        super().__init__(*args, **kwargs)
        self.CatClass = SparseCategorical