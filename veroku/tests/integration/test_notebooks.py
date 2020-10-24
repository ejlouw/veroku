from importnb import Notebook
import unittest


class TestSudoku(unittest.TestCase):
    """
    A test class for example notebooks
    """
    def test_sudoku(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import veroku.notebooks.sudoku_example

    def test_slip_on_grass(self):
        """
        Test that the sudoku notebook runs successfully and computes the correct solution (checked in notebook)
        :return:
        """
        with Notebook():
            import veroku.notebooks.slip_on_grass_example