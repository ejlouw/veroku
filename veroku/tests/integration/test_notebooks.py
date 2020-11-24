from mockito import when, unstub
from importnb import Notebook
import unittest
from veroku.cluster_graph import ClusterGraph
import numpy as np
import sys
import os


class TestNotebooks(unittest.TestCase):
    """
    A test class for example notebooks
    """
    def setUp(self):
        """
        Run before every test.
        """
        when(ClusterGraph).show().thenReturn()
        print('                 pwd: ', os.getcwd())
        sys.path.append('./examples')
        sys.path.append('../examples')
        sys.path.append('../../examples')
        sys.path.append('../../../examples')

    def tearDown(self):
        """
        Run after every test.
        """
        unstub()

    def test_sudoku(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution.
        :return:
        """
        with Notebook():
            import examples.sudoku
            infered_solution_array = examples.sudoku.infered_solution_array
            correct_solution_array = examples.sudoku.correct_solution_array
            self.assertTrue(np.array_equal(infered_solution_array, correct_solution_array))

    def test_slip_on_grass(self):
        """
        Test that the slip_on_grass notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import examples.slip_on_grass

    def test_kalman_filter(self):
        """
        Test that the Kalman filter notebook runs successfully and computes the correct solution
        """

        with Notebook():
            import examples.Kalman_filter

            position_posteriors = examples.Kalman_filter.position_posteriors
            factors = examples.Kalman_filter.factors
            evidence_dict = examples.Kalman_filter.evidence_dict

            marginal_vars = [p.var_names for p in position_posteriors]
            joint = factors[0]
            for f in factors[1:]:
                joint = joint.absorb(f)

            joint = joint.reduce(vrs=list(evidence_dict.keys()),
                                 values=list(evidence_dict.values()))
            correct_marginals = []
            for vrs in marginal_vars:
                correct_marginal = joint.marginalize(vrs, keep=True)
                correct_marginals.append(correct_marginal)
            for actual_marginal, expected_marginal in zip(position_posteriors, correct_marginals):
                actual_K = actual_marginal.get_K()
                expected_K = expected_marginal.get_K()
                actual_h = actual_marginal.get_h()
                expected_h = expected_marginal.get_h()
                # TODO: see why log_weight (and g) parameters are so different between the actual and expected factors.
                self.assertTrue(np.allclose(actual_K, expected_K))
                self.assertTrue(np.allclose(actual_h, expected_h, rtol=1.e-5, atol=1.e-5))
