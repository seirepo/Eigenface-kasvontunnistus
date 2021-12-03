import unittest
import numpy as np
from individual import Individual
import operations as op

class TestIndividual(unittest.TestCase):
    def setUp(self):
        self.images = np.array( [[ 1, 2, 3],
                                 [ 1, 1, 1],
                                 [ 2, 0, 1]] )

    def test_constructor_creates_individual_correctly(self):
        ind = Individual(2, self.images)
        self.assertTrue((ind.images == self.images).all())
        self.assertEqual(ind.id, 2)

    def test_select_average_returns_correct_shape(self):
        ind = Individual(2, self.images)
        self.assertEqual(ind.average_images.shape, (3, 2))

    def test_select_average_returns_correct_array(self):
        ims = np.hstack((self.images, np.array([1,1,1]).reshape((3,1))))
        ind = Individual(2, ims)
        result = ind.average_images
        should_be = np.array([[1.5, 1, 1], [2, 1, 1]]).T

        self.assertTrue((result == should_be).all())