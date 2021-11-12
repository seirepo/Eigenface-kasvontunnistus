import unittest
import numpy as np
from individual import Individual

class TestIndividual(unittest.TestCase):
    def setUp(self):
        self.m1 = np.array( [[ 1, 2, 3],
                            [ 4, 5, 6]])
        self.m2 = np.array( [[ 7, 8, 9],
                            [ 10, 11, 12]])
        self.m3 = np.array( [[ 13, 14, 15],
                            [ 16, 17, 18]])
        self.m4 = np.array( [[ 19, 20],
                            [ 21, 22]])
        self.empty = np.array([[1]])


    def test_matrix2d_to_column_vector_returns_array_of_right_shape(self):
        result1 = Individual.matrix2d_to_column_vector(self.m1)
        result2 = Individual.matrix2d_to_column_vector(self.m4)
        result3 = Individual.matrix2d_to_column_vector(self.empty)
        self.assertEqual(result1.shape, (6, 1))
        self.assertEqual(result2.shape, (4, 1))
        self.assertEqual(result3.shape, (1, 1))

    def test_matrix2d_to_column_vector_returns_correct_vector(self):
        result1 = Individual.matrix2d_to_column_vector(self.m1)
        result2 = Individual.matrix2d_to_column_vector(self.m4)
        result3 = Individual.matrix2d_to_column_vector(self.empty)
        self.assertTrue((result1 == np.array([[1, 2, 3, 4, 5, 6]]).T).all())
        self.assertFalse((result1 == np.array([[1, 2, 3, 4, 5, 6]])).all())
        self.assertTrue((result2 == np.array([[19, 20, 21, 22]]).T).all())
        self.assertTrue((result3 == np.array([[1]])).all())

    def test_matrix_to_vectors_returns_matrix_with(self):
        test_matrix = np.array([ [self.m1, self.m2, self.m3] ])
        result = Individual.matrix_to_vectors(test_matrix)
        # should be "np.array( [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
        #                      [13, 14, 15, 16, 17, 18]] )"
        self.assertEqual(result.shape, (6, 3))
        self.assertEqual(result[0].tolist(), np.array([[1, 2, 3, 4, 5, 6]]).tolist())
        self.assertEqual(result[1].tolist(), np.array([[7, 8, 9, 10, 11, 12]]).tolist())
        self.assertEqual(result[2].tolist(), np.array([[13, 14, 15, 16, 17, 18]]).tolist())
