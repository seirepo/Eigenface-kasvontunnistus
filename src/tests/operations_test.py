import unittest
import numpy as np
import operations as op

class TestOperations(unittest.TestCase):
    def setUp(self):
        self.mat1 = np.array( [[ 1, 2, 3],
                            [ 4, 5, 6]])
        self.mat2 = np.array( [[ 7, 8, 9],
                            [ 10, 11, 12]])
        self.mat3 = np.array( [[ 13, 14, 15],
                            [ 16, 17, 18]])
        self.mat4 = np.array( [[ 19, 20],
                            [ 21, 22]])
        self.mat5 = np.array([[1]])

    def test_matrix_submatrices_to_colums_returns_matrix_with_correct_size(self):
        test_matrix = np.array([ self.mat1, self.mat2, self.mat3 ])
        result = op.matrix3d_submatrices_to_colums(test_matrix)
        self.assertEqual(result.shape, (6, 3))

    def test_matrix_submatrices_to_colums_returns_matrix_with_submatrices_as_columns(self):
        test_matrix = np.array([ self.mat1, self.mat2, self.mat3 ])
        result = op.matrix3d_submatrices_to_colums(test_matrix)
        should_be = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]]).T

        self.assertEqual(result.shape, should_be.shape)
        self.assertTrue((result == should_be).all())
