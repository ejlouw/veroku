"""
Tests for the Gaussian module.
"""

# System imports
import unittest
import cmath

# Third-party imports
from mockito import when, expect, unstub, verifyNoUnwantedInteractions
import numpy as np
from scipy import integrate

# Local imports
from veroku.factors.gaussian import Gaussian
from veroku.factors import _factor_utils


# pylint: disable=too-many-public-methods
class TestGaussian(unittest.TestCase):
    """
    Tests for Gaussian class.
    """

    def test_constructor_insufficient_parameters(self):
        """
        Test that the constructor raises an exception when insufficient parameters are supplied
        """
        # no var_names
        with self.assertRaises(Exception):
            Gaussian()
        # incomplete covariance form parameters
        with self.assertRaises(ValueError):
            Gaussian(cov=5, var_names=['a'])
        # incomplete canonical form parameters
        with self.assertRaises(ValueError):
            Gaussian(K=5, var_names=['a'])

    def test_constructor_variable_fail(self):
        """
        Test that the constructor raises a value error when the variables are not unique.
        """
        with self.assertRaises(ValueError):
            Gaussian(cov=[[1, 0], [0, 1]], mean=[1, 1], log_weight=0.0, var_names=['a', 'a'])

    def test_covariance_form_constructor(self):
        """
        Test that the constructor constructs an object with the correct covariance parameters
        """
        cov_mat_list = [[5.0, 1.0], [1.0, 5.0]]
        cov_mat_array = np.array(cov_mat_list)
        mean_vec_list = [[2.0], [4.0]]
        mean_vec_array = np.array(mean_vec_list)
        when(_factor_utils).make_square_matrix(cov_mat_list).thenReturn(cov_mat_array)
        when(_factor_utils).make_column_vector(mean_vec_list).thenReturn(mean_vec_array)

        gaussian_a = Gaussian(cov=cov_mat_list, mean=mean_vec_list, log_weight=0.0, var_names=['a', 'b'])
        self.assertEqual(gaussian_a.var_names, ['a', 'b'])
        self.assertTrue(np.array_equal(gaussian_a.cov, cov_mat_array))
        self.assertTrue(np.array_equal(gaussian_a.mean, mean_vec_array))
        self.assertEqual(gaussian_a.log_weight, 0.0)
        self.assertEqual(gaussian_a.K, None)
        self.assertEqual(gaussian_a.h, None)
        self.assertEqual(gaussian_a.g, None)
        unstub()

    def test_canonical_form_constructor(self):
        """
        Test that the constructor constructs an object with the correct coninical parameters
        """
        # pylint: disable=invalid-name
        K_mat_list = [[5.0, 1.0], [1.0, 5.0]]
        K_mat_array = np.array(K_mat_list)
        h_vec_list = [[2.0], [4.0]]
        h_vec_array = np.array(h_vec_list)
        when(_factor_utils).make_square_matrix(K_mat_list).thenReturn(K_mat_array)
        when(_factor_utils).make_column_vector(h_vec_list).thenReturn(h_vec_array)

        gaussian_a = Gaussian(K=K_mat_list, h=h_vec_list, g=0.0, var_names=['a', 'b'])
        # pylint: enable=invalid-name
        self.assertEqual(gaussian_a.var_names, ['a', 'b'])
        self.assertTrue(np.array_equal(gaussian_a.K, K_mat_array))
        self.assertTrue(np.array_equal(gaussian_a.h, h_vec_array))
        self.assertEqual(gaussian_a.g, 0.0)
        self.assertEqual(gaussian_a.cov, None)
        self.assertEqual(gaussian_a.mean, None)
        self.assertEqual(gaussian_a.log_weight, None)
        unstub()

    def test_get_cov(self):
        """
        Test that the correct covariance is returned.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        self.assertTrue(np.array_equal(gaussian_a.get_cov(), np.array([[2.0]])))

    @staticmethod
    def test_get_cov_from_canform():
        """
        Test that the _update_covform function is called before returning the covariance parameter.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        # pylint: disable=protected-access
        expect(Gaussian, times=1)._update_covform()
        # pylint: enable=protected-access
        gaussian_a.get_cov()
        verifyNoUnwantedInteractions()
        unstub()

    def test_get_mean(self):
        """
        Test that the correct mean is returned.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        self.assertTrue(np.array_equal(gaussian_a.get_mean(), np.array([[1.0]])))

    @staticmethod
    def test_get_mean_from_canform():
        """
        Test that the _update_covform function is called before returning the mean parameter.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        # pylint: disable=protected-access
        expect(Gaussian, times=1)._update_covform()
        # pylint: enable=protected-access
        gaussian_a.get_mean()
        verifyNoUnwantedInteractions()
        unstub()

    # pylint: disable=protected-access
    def test_get_log_weight(self):
        """
        Test that the correct log-weight is returned.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        self.assertEqual(gaussian_a.get_log_weight(), 0.0)

    @staticmethod
    def test_get_log_weight_from_canform():
        """
        Test that the _update_covform function is called before returning the log-weight parameter.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        expect(Gaussian, times=1)._update_covform()
        gaussian_a.get_log_weight()
        verifyNoUnwantedInteractions()
        unstub()

    # pylint: disable=invalid-name
    def test_get_K(self):
        """
        Test that the correct K is returned.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        self.assertTrue(np.array_equal(gaussian_a.get_K(), np.array([[2.0]])))

    # pylint: enable=invalid-name

    # pylint: disable=invalid-name
    @staticmethod
    def test_get_K_from_covform():
        """
        Test that the _update_canform function is called before returning the K parameter.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        # pylint: disable=protected-access
        expect(Gaussian, times=1)._update_canform()
        # pylint: enable=protected-access
        gaussian_a.get_K()
        verifyNoUnwantedInteractions()
        unstub()

    # pylint: enable=invalid-name

    def test_impossible_update_canform(self):
        """
        Test that the _update_canform raises a LinAlgError when the covariance matrix is not invertible.
        """
        gaussian = Gaussian(cov=np.zeros([2, 2]), mean=[0.0, 0.0], log_weight=0.0, var_names=['a', 'b'])
        with self.assertRaises(np.linalg.LinAlgError):
            # pylint: disable=protected-access
            gaussian._update_canform()
            # pylint: enable=protected-access

    def test_impossible_update_covform(self):
        """
        Test that the _update_covform raises a LinAlgError when the precision matrix is not invertible.
        """
        gaussian = Gaussian(K=np.zeros([2, 2]), h=[0.0, 0.0], g=0.0, var_names=['a', 'b'])
        with self.assertRaises(Exception):
            # pylint: disable=protected-access
            gaussian._update_covform()
            # pylint: enable=protected-access

    def test_get_h(self):
        """
        Test that the correct h is returned.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        self.assertTrue(np.array_equal(gaussian_a.get_h(), np.array([[1.0]])))

    @staticmethod
    def test_get_h_from_covform():
        """
        Test that the _update_canform function is called before returning the h parameter.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        # pylint: disable=protected-access
        expect(Gaussian, times=1)._update_canform()
        # pylint: enable=protected-access
        gaussian_a.get_h()
        verifyNoUnwantedInteractions()
        unstub()

    def test_get_g(self):
        """
        Test that the correct g is returned.
        """
        gaussian_a = Gaussian(K=2.0, h=1.0, g=0.0, var_names=['a'])
        self.assertEqual(gaussian_a.get_g(), 0.0)

    # pylint: disable=protected-access
    @staticmethod
    def test_get_g_from_covform():
        """
        Test that the _update_canform function is called before returning the g parameter.
        """
        gaussian_a = Gaussian(cov=2.0, mean=1.0, log_weight=0.0, var_names=['a'])
        expect(Gaussian, times=1)._update_canform()
        gaussian_a.get_g()
        verifyNoUnwantedInteractions()
        unstub()

    # pylint: enable=protected-access

    def test_absorb_1d(self):
        """
        Test that the Gaussian multiplication function returns the correct result for one dimensional Gaussians.
        """
        gaussian_a = Gaussian(K=5.0, h=4.0, g=3.0, var_names=['a'])
        gaussian_b = Gaussian(K=3.0, h=2.0, g=1.0, var_names=['a'])
        expected_product = Gaussian(K=8.0, h=6.0, g=4.0, var_names=['a'])
        actual_product = gaussian_a.absorb(gaussian_b)
        self.assertTrue(expected_product.equals(actual_product))

    def test_cancel_1d(self):
        """
        Test that the Gaussian division function returns the correct result for one dimensional Gaussians.
        """
        gaussian_a = Gaussian(K=6.0, h=4.0, g=2.0, var_names=['a'])
        gaussian_b = Gaussian(K=3.0, h=2.0, g=1.0, var_names=['a'])
        expected_quotient = Gaussian(K=3.0, h=2.0, g=1.0, var_names=['a'])
        actual_quotient = gaussian_a.cancel(gaussian_b)
        self.assertTrue(expected_quotient.equals(actual_quotient))

    def test_absorb_2d(self):
        """
        Test that the Gaussian multiplication function returns the correct result for two dimensional Gaussians.
        """
        gaussian_a = Gaussian(K=[[5.0, 2.0], [2.0, 6.0]], h=[1.0, 2.0], g=3.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[4.0, 1.0], [1.0, 4.0]], h=[2.0, 3.0], g=2.0, var_names=['a', 'b'])
        expected_product = Gaussian(K=[[9.0, 3.0], [3.0, 10.0]], h=[3.0, 5.0], g=5.0, var_names=['a', 'b'])
        actual_product = gaussian_a.absorb(gaussian_b)
        self.assertTrue(expected_product.equals(actual_product))

    def test_cancel_2d(self):
        """
        Test that the Gaussian division function returns the correct result for two dimensional Gaussians.
        """
        gaussian_a = Gaussian(K=[[7.0, 2.0], [2.0, 6.0]], h=[4.0, 3.0], g=3.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[4.0, 1.0], [1.0, 4.0]], h=[1.0, 2.0], g=2.0, var_names=['a', 'b'])

        expected_quotient = Gaussian(K=[[3.0, 1.0], [1.0, 2.0]], h=[3.0, 1.0], g=1.0, var_names=['a', 'b'])
        actual_quotient = gaussian_a.cancel(gaussian_b)
        self.assertTrue(expected_quotient.equals(actual_quotient))

        gaussian_a_reordered = Gaussian(K=[[6.0, 2.0], [2.0, 7.0]], h=[3.0, 4.0], g=3.0, var_names=['b', 'a'])
        actual_quotient = gaussian_a_reordered.cancel(gaussian_b)
        # pylint: disable=protected-access
        actual_quotient._reorder_parameters(['a', 'b'])
        # pylint: enable=protected-access
        self.assertTrue(expected_quotient.equals(actual_quotient))

    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals
    def test_observe(self):
        """
        Test that the Gaussian observe function returns the correct result.
        """
        # Note these equations where written independently from the actuall implementation.
        # TODO: consider extending this test and hard-coding the expected parmeters
        # TODO: Fix this test - it passes sometimes and sometimes not - something to do with the paramter order.

        Kmat = np.array([[6, 2, 1],
                         [2, 8, 3],
                         [1, 3, 9]])
        hvec = np.array([[1],
                         [2],
                         [3]])
        g_val = 1.5

        # The block-matrix sub-parameters
        K_xx = np.array([[6, 2],
                         [2, 8]])
        K_xy = np.array([[1],
                         [3]])
        K_yy = np.array([[9]])

        h_x = np.array([[1],
                        [2]])
        h_y = np.array([[3]])
        z_observed = np.array([[6]])

        expected_K = K_xx
        expeted_h = h_x - K_xy.dot(z_observed)
        expected_g = g_val + h_y.transpose().dot(z_observed) - 0.5 * z_observed.transpose().dot(K_yy).dot(z_observed)
        expected_gaussian = Gaussian(K=expected_K, h=expeted_h, g=expected_g, var_names=['x', 'y'])
        gaussian = Gaussian(K=Kmat, h=hvec, g=g_val, var_names=['x', 'y', 'z'])
        actual_gaussian = gaussian.observe(vrs=['z'], values=z_observed)
        expected_gaussian.show()
        actual_gaussian.show()
        self.assertTrue(actual_gaussian.equals(expected_gaussian))

        # Test that the result is still correct with a different parameter order.
        # pylint: disable=protected-access
        gaussian_copy = gaussian.copy()
        gaussian_copy._reorder_parameters(['x', 'z', 'y'])
        # pylint: enable=protected-access
        actual_gaussian = gaussian_copy.observe(vrs=['z'], values=z_observed)
        self.assertTrue(actual_gaussian.equals(expected_gaussian))

    # pylint: enable=too-many-locals
    # pylint: enable=invalid-name

    def test_reorder_parameters(self):
        """
        Test that _reorder_parameters properly reorders the values in the canonical parameters.
        :return:
        """
        gaussian_a = Gaussian(K=[[1.0, 2.0], [2.0, 3.0]], h=[1.0, 2.0], g=1.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[3.0, 2.0], [2.0, 1.0]], h=[2.0, 1.0], g=1.0, var_names=['b', 'a'])
        # pylint: disable=protected-access
        gaussian_b._reorder_parameters(['a', 'b'])
        # pylint: enable=protected-access
        self.assertTrue(gaussian_a.equals(gaussian_b))

        gaussian_a = Gaussian(cov=[[1.0, 2.0], [2.0, 3.0]], mean=[1.0, 2.0], log_weight=1.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(cov=[[3.0, 2.0], [2.0, 1.0]], mean=[2.0, 1.0], log_weight=1.0, var_names=['b', 'a'])
        # pylint: disable=protected-access
        gaussian_b._reorder_parameters(['a', 'b'])
        # pylint: enable=protected-access
        self.assertTrue(gaussian_a.equals(gaussian_b))

    def test_marginalise_2d(self):
        """
        Test that the Gaussian marginalisation function returns the correct result for a two dimensional Gaussians.
        :return:
        """
        gaussian = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        expected_result = Gaussian(cov=7.0, mean=4.0, log_weight=0.0, var_names=['a'])
        actual_result = gaussian.marginalise(vrs=['a'], keep=True)
        self.assertTrue(expected_result.equals(actual_result))

    def test_marginalise_2d_canform(self):
        """
        Test that the Gaussian marginalisation function returns the correct result for a two dimensional Gaussians.
        :return:
        """
        gaussian = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        expected_result = Gaussian(cov=7.0, mean=4.0, log_weight=0.0, var_names=['a'])
        expected_result._update_canform()
        gaussian._update_canform()
        actual_result = gaussian.marginalise(vrs=['a'], keep=True)
        self.assertTrue(expected_result.equals(actual_result))

    def test_equals(self):
        """
        Test that the equals function can identify Gaussians that differ only in their variables.
        """
        gaussian_1 = Gaussian(cov=1, mean=1, log_weight=1.0, var_names=['a'])
        gaussian_2 = Gaussian(cov=0, mean=1, log_weight=1.0, var_names=['b'])
        self.assertFalse(gaussian_1.equals(gaussian_2))

    def test_equals_different_order(self):
        """
        Test that the equals function can identify identical Gaussians with different order variables.
        """
        gaussian_1 = Gaussian(cov=[[1, 0], [0, 2]], mean=[1, 2], log_weight=1.0, var_names=['a', 'b'])
        gaussian_2 = Gaussian(cov=[[2, 0], [0, 1]], mean=[2, 1], log_weight=1.0, var_names=['b', 'a'])

        self.assertTrue(gaussian_1.equals(gaussian_2))

    def test_covform_equals(self):
        """
        Test that the _covform_equals function returns false if any of the covariance form parameters differ.
        """
        gaussian_1 = Gaussian(cov=1, mean=1, log_weight=1.0, var_names=['a'])
        gaussian_2 = Gaussian(cov=0, mean=1, log_weight=1.0, var_names=['a'])
        # pylint: disable=protected-access
        # with different covariances
        self.assertFalse(gaussian_1._covform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # with different means
        gaussian_2 = Gaussian(cov=1, mean=0, log_weight=1.0, var_names=['a'])
        self.assertFalse(gaussian_1._covform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # with different log_weights
        gaussian_2 = Gaussian(cov=1, mean=1, log_weight=0.0, var_names=['a'])
        self.assertFalse(gaussian_1._covform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # pylint: enable=protected-access

    def test_canform_equals(self):
        """
        Test that the _canform_equals function returns false if any of the canonical form parameters differ.
        """
        gaussian_1 = Gaussian(K=1, h=1, g=1.0, var_names=['a'])
        gaussian_2 = Gaussian(K=0, h=1, g=1.0, var_names=['a'])
        # pylint: disable=protected-access
        # with different Ks
        self.assertFalse(gaussian_1._canform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # with different hs
        gaussian_2 = Gaussian(K=1, h=0, g=1.0, var_names=['a'])
        self.assertFalse(gaussian_1._canform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # with different gs
        gaussian_2 = Gaussian(K=1, h=1, g=0.0, var_names=['a'])
        self.assertFalse(gaussian_1._canform_equals(gaussian_2, rtol=0.0, atol=0.0))
        # pylint: enable=protected-access

    def test_equals_covform_true(self):
        """
        Test that the equals function can identify identical and effectively identical Gaussians in covariance form.
        """
        gaussian = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        same_gaussian = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(same_gaussian))

        # Test approximately equals
        error = 1e-7
        effectively_same_gaussian = Gaussian(cov=[[7.0 + error, 2.0 + error], [2.0 + error, 1.0 + error]],
                                             mean=[4.0 + error, 1.0 + error],
                                             log_weight=0.0 + error, var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(effectively_same_gaussian))

    def test_equals_canform_true(self):
        """
        Test that the equals function can identify identical and effectively identical Gaussians in covariance form.
        """
        gaussian = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        same_gaussian = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(same_gaussian))

        # Test approximately equals
        error = 1e-7
        effectively_same_gaussian = Gaussian(K=[[7.0 + error, 2.0 + error], [2.0 + error, 1.0 + error]],
                                             h=[4.0 + error, 1.0 + error], g=0.0 + error,
                                             var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(effectively_same_gaussian))

    def test_equals_false_covform(self):
        """
        Test that the equals function can identify different Gaussians (with differences in the
        various covariance form parameters).
        """
        gaussian_a = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(cov=[[2.0, 1.0], [1.0, 2.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

        gaussian_a = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[0.0, 0.0], log_weight=0.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

        gaussian_a = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=1.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

    def test_equals_false_canform(self):
        """
        Test that the equals function can identify different Gaussians (with differences in the
        various canonical form parameters).
        """
        gaussian_a = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[2.0, 1.0], [1.0, 2.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

        gaussian_a = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[0.0, 0.0], g=0.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

        gaussian_a = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        gaussian_b = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=1.0, var_names=['a', 'b'])
        self.assertFalse(gaussian_a.equals(gaussian_b))

    def test_copy_no_form(self):
        """
        Test that the copy function raises an exception when a Gaussian does not have either of its form updated.
        """
        gaussian_no_form = Gaussian(cov=1.0, mean=0.0, log_weight=0.0, var_names=['a'])
        gaussian_no_form.COVFORM = False
        with self.assertRaises(Exception):
            gaussian_no_form.copy()

    def test_copy_1d_covform(self):
        """
        Test that the copy function returns a identical copy of a one dimensional Gaussian in covariance form.
        """
        gaussian = Gaussian(cov=7.0, mean=4.0, log_weight=0.0, var_names=['a'])
        self.assertTrue(gaussian.equals(gaussian.copy()))

    def test_copy_2d_covform(self):
        """
        Test that the copy function returns a identical copy of a two dimensional Gaussian in covariance form.
        """
        gaussian = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(gaussian.copy()))

    def test_copy_1d_canform(self):
        """
        Test that the copy function returns a identical copy of a Gaussian in canonical form.
        """
        gaussian = Gaussian(K=7.0, h=4.0, g=0.0, var_names=['a'])
        self.assertTrue(gaussian.equals(gaussian.copy()))

    def test_copy_2d_canform(self):
        """
        Test that the copy function returns a identical copy of a Gaussian in canonical form.
        """
        gaussian = Gaussian(K=[[7.0, 2.0], [2.0, 1.0]], h=[4.0, 1.0], g=0.0, var_names=['a', 'b'])
        self.assertTrue(gaussian.equals(gaussian.copy()))

    def test_form_conversion(self):
        """
        Test that conversion from one form to the other and back results in the same Gaussian parameters
        :return:
        """
        gaussian_ab = Gaussian(cov=[[7.0, 2.0], [2.0, 1.0]], mean=[4.0, 1.0], log_weight=0.0, var_names=['a', 'b'])
        gaussian_ab_copy = gaussian_ab.copy()
        # pylint: disable=protected-access
        gaussian_ab_copy._update_canform()
        gaussian_ab_copy.COVFORM = False
        gaussian_ab_copy._update_covform()
        # pylint: enable=protected-access
        self.assertTrue(gaussian_ab.equals(gaussian_ab_copy))

    def test_get_complex_log_weight(self):
        """
        Test that the _get_complex_log_weight returns the correct value.
        """
        gaussian_ab = Gaussian(K=[[-1.0, 0.0], [0.0, 1.0]], h=[0.0, 0.0], g=0.0, var_names=['a', 'b'])
        actual_complex_weight = gaussian_ab.get_complex_weight()
        expected_complex_weight = cmath.exp(.5 * cmath.log((-1.0) * (2.0 * np.pi) ** 2))
        self.assertAlmostEqual(actual_complex_weight, expected_complex_weight)

    def test_invert(self):
        """
        Test that the _invert method returns a Gaussian with the negated parameters.
        :return:
        """
        gaussian = Gaussian(K=[[1.0, 2.0], [2.0, 3.0]], h=[4.0, 5.0], g=6.0, var_names=['a', 'b'])
        expected_inv_gaussian = Gaussian(K=[[-1.0, -2.0], [-2.0, -3.0]], h=[-4.0, -5.0], g=-6.0, var_names=['a', 'b'])
        actual_inv_gaussian = gaussian._invert()
        self.assertTrue(actual_inv_gaussian.equals(expected_inv_gaussian))

    def test_weight_and_integration_1d(self):
        """
        Test that the Gaussian distribution intagrates to the weight.
        """
        weight = 1.0
        gaussian = Gaussian(cov=2.0, mean=4.0, log_weight=np.log(weight), var_names=['a'])
        definite_integral, _ = integrate.quad(gaussian.potential, -100.0, 100.0)
        self.assertAlmostEqual(definite_integral, weight)

        # test another weight and with canonical form
        weight = 2.0
        gaussian = Gaussian(cov=2.0, mean=4.0, log_weight=np.log(weight), var_names=['a'])
        # pylint: disable=protected-access
        gaussian._update_canform()
        # pylint: enable=protected-access
        gaussian.COVFORM = False
        definite_integral, _ = integrate.quad(gaussian.potential, -100.0, 100.0)
        self.assertAlmostEqual(definite_integral, weight)

    def test_log_potential_no_form(self):
        """
        Test that the log_potential function raises an exception if the Gaussian has no form.
        """
        gaussian = Gaussian(cov=1.0, mean=1.0, log_weight=np.log(1.0), var_names=['a'])
        gaussian.COVFORM = False
        with self.assertRaises(Exception):
            gaussian.log_potential(x_val=0)

    def test_weight_and_integration_2d(self):
        """
        Test that the Gaussian distribution intagrates to the weight.
        """
        weight = 1.0
        gaussian = Gaussian(cov=[[2.0, 1.0], [1.0, 2.0]], mean=[4.0, 5.0], log_weight=np.log(weight),
                            var_names=['a', 'b'])
        definite_integral, _ = integrate.dblquad(lambda x1, x2: gaussian.potential([x1, x2]), -20.0, 20.0,
                                                 lambda x: -20.0, lambda x: 20.0)
        self.assertAlmostEqual(definite_integral, weight)
        weight = 2.0
        gaussian = Gaussian(cov=[[3.0, 2.0], [2.0, 6.0]], mean=[-4.0, 5.0], log_weight=np.log(weight),
                            var_names=['a', 'b'])
        definite_integral, _ = integrate.dblquad(lambda x1, x2: gaussian.potential([x1, x2]), -20.0, 20.0,
                                                 lambda x: -20.0, lambda x: 20.0)
        self.assertAlmostEqual(definite_integral, weight)

    def test_kl_divergence_between_vacuous_factors(self):
        """
        Test that the distance between two vacuous factors is zero.
        """
        g1 = Gaussian.make_vacuous(var_names=['a', 'b'])
        g2 = Gaussian.make_vacuous(var_names=['a', 'b'])
        self.assertTrue(g1.kl_divergence(g2) == 0.0)

    def test_vacuous_equals(self):
        g1 = Gaussian.make_vacuous(var_names=['a', 'b'])
        g2 = Gaussian.make_vacuous(var_names=['a', 'b'])
        self.assertTrue(g1.equals(g2))

    def test_log_potential(self):
        gaussian = Gaussian(K=[[7.0, 2.0], [2.0, 6.0]], h=[4.0, 3.0], g=0.0, var_names=['a', 'b'])


        x_val = [7, 3]
        vrs = ['a', 'b']
        log_pot_canform = gaussian.log_potential(x_val=x_val, vrs=vrs)
        gaussian._update_covform()
        gaussian.CANFORM = False
        log_pot_covform = gaussian.log_potential(x_val=x_val, vrs=vrs)
        # TODO: investigate the
        self.assertAlmostEqual(log_pot_canform, log_pot_covform, places=1)

    # TODO: add tests for vacuous Gaussians.
# pylint: enable=too-many-public-methods
