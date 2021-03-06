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
        self.rng = np.random.default_rng(seed=42)

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
        self.assertEqual(res.shape, (2,))

    def test_get_coordinates_in_given_base_returns_correct_coordinates(self):
        im = np.array([3,0,-2])
        space = np.array([[1,0,0],[0,1,0],[0,0,1]])
        av = np.zeros(im.shape)
        result = op.get_coordinates(im, space, av)
        should_be = np.array([3,0,-2])

        self.assertTrue((result == should_be).all())

    def test_get_coordinates_returns_array_of_correct_shape(self):
        rndm = self.rng.random((4096, 320))
        basis = np.linalg.qr(rndm)[0]
        image = self.rng.random((4096))
        av = np.zeros(image.shape)

        result = op.get_coordinates(image, basis, av)
        print(basis.shape)
        r = np.dot(image, basis[:,0])
        e = r * basis[:,0]

        self.assertEqual(result.shape, (320,))

    def test_get_most_frequent_returns_most_frequent_value(self):
        l1 = [1, 1, 1]
        l2 = [12, 22, 22]
        l3 = [39, 8, 4]
        l4 = [34, 4, 4, 20]
        l5 = [9, 7, 0, 7]
        l6 = [39, 4, 39, 4]

        self.assertEqual(op.get_most_frequent(l1), 1)
        self.assertEqual(op.get_most_frequent(l2), 22)
        self.assertEqual(op.get_most_frequent(l3), 39)
        self.assertEqual(op.get_most_frequent(l4), 4)
        self.assertEqual(op.get_most_frequent(l5), 7)
        self.assertEqual(op.get_most_frequent(l6), 39)
