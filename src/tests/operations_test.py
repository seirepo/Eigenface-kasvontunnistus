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
        self.test_matrix = np.array([ self.mat1, self.mat2, self.mat3 ])

    def test_matrix_submatrices_to_columns_returns_matrix_with_correct_size(self):
        result = op.images_to_vectors(self.test_matrix)
        self.assertEqual(result.shape, (6, 3))

    def test_matrix_submatrices_to_columns_returns_matrix_with_submatrices_as_columns(self):
        result = op.images_to_vectors(self.test_matrix)
        should_be = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]]).T

        self.assertEqual(result.shape, should_be.shape)
        self.assertTrue((result == should_be).all())

    def test_matrix_submatrices_to_columns_raises_error(self):
        self.assertRaises(ValueError,
            op.images_to_vectors, self.mat1)
        self.assertRaises(ValueError,
            op.images_to_vectors, np.array([self.test_matrix]))

    def test_calculate_eigenface_returns_eigenface_matrix_of_correct_shape(self):
        #result1 = op.calculate_eigenfaces(self.mat1)
        result2 = op.calculate_eigenfaces(self.mat1, 1)
        #result3 = op.calculate_eigenfaces(self.mat1, 0)
        result4 = op.calculate_eigenfaces(self.mat4, 2)
        #result5 = op.calculate_eigenfaces(self.mat5)
        #result6 = op.calculate_eigenfaces(self.mat5, -5)

        #self.assertEqual(result1.shape, (2, 3))
        self.assertEqual(result2.shape, (2, 1))
        #self.assertEqual(result3.shape, (2, 3))
        self.assertEqual(result4.shape, (2, 2))
        #self.assertEqual(result5.shape, (1, 1))
        #self.assertEqual(result6.shape, result5.shape)

    def test_calculate_eigenfaces_throws_exception(self):
        self.assertRaises(Exception,
            op.calculate_eigenfaces, self.mat1, 15)
        self.assertRaises(Exception,
            op.calculate_eigenfaces, self.mat4, 3)

    def test_get_average_face_returns_average_correctly(self):
        res = op.get_average_face(self.mat1)
        self.assertEqual(res.shape, (2, 1))
        #self.assertAlmostEqual((res == np.array([2, 5])).all())