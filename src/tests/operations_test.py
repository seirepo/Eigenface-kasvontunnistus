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
        result2 = op.calculate_eigenfaces(self.mat1, 1)
        result4 = op.calculate_eigenfaces(self.mat4, 2)

        self.assertEqual(result2.shape, (2, 1))
        self.assertEqual(result4.shape, (2, 2))

    def test_calculate_eigenfaces_throws_exception(self):
        self.assertRaises(Exception,
            op.calculate_eigenfaces, self.mat1, 15)
        self.assertRaises(Exception,
            op.calculate_eigenfaces, self.mat4, 3)

    def test_get_average_face_returns_average_correctly(self):
        res = op.get_average_face(self.mat1)
        self.assertEqual(res.shape, (2, 1))

    def test_get_eigenfaces(self):
        eigvecs = np.array([[2,1,1],[3,0,1],[2,3,2]])
        images = np.array([[1,2,3],[1,5,6],[0,4,2]])
        result = op.get_eigenfaces(images, eigvecs)
        should_be = np.array([[14,10,9],[29,19,18],[16,6,8]])

        self.assertEqual(result.shape, (3, 3))
        self.assertTrue((result == should_be).all())

    def test_get_coordinates_in_given_base_returns_correct_coordinates(self):
        im = np.array([3,0,-2])
        space = np.array([[1,0,0],[0,1,0],[0,0,1]])
        result = op.get_coordinates(im, space)
        should_be = np.array([3,0,-2])

        self.assertTrue((result == should_be).all())
