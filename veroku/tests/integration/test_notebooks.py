from mockito import when, unstub, ANY
from importnb import Notebook
import unittest
from veroku.notebooks import kalman_filter_plotting
from veroku.cluster_graph import ClusterGraph
import numpy as np


class TestSudoku(unittest.TestCase):
    """
    A test class for example notebooks
    """
    def setUp(self):
        """
        Run before every test.
        """
        when(ClusterGraph).show().thenReturn()

    def tearDown(self):
        """
        Run after every test.
        """
        unstub()

    def test_sudoku(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import veroku.notebooks.sudoku_example
            infered_solution_array = veroku.notebooks.sudoku_example.infered_solution_array
            correct_solution_array = veroku.notebooks.sudoku_example.correct_solution_array
            veroku.notebooks.sudoku_example
            self.assertTrue(np.array_equal(infered_solution_array, correct_solution_array))

    def test_slip_on_grass(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import veroku.notebooks.slip_on_grass_example

    def test_kalman_filter(self):
        """
        Test that the Kalman filter notebook runs successfully and computes the correct solution
        """

        with Notebook() as nb:
            when(kalman_filter_plotting).infered_system_state_widget(ANY, ANY, ANY).thenReturn()
            import veroku.notebooks.Kalman_filter

            position_posteriors = veroku.notebooks.Kalman_filter.position_posteriors
            factors = veroku.notebooks.Kalman_filter.factors
            evidence_dict = veroku.notebooks.Kalman_filter.evidence_dict

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
                self.assertTrue(actual_marginal.equals(expected_marginal, rtol=1e-01, atol=1e-01))
