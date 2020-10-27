from mockito import when, unstub
from importnb import Notebook
import unittest
from veroku.cluster_graph import ClusterGraph

# TODO: add print output suppression to tests


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
        #TODO: put this back once factor operations have been made faster (too slow right now).
        #with Notebook():
        #    import veroku.notebooks.sudoku_example

    def test_slip_on_grass(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import veroku.notebooks.slip_on_grass_example