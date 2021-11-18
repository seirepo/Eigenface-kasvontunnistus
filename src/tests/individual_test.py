import unittest
import numpy as np
from individual import Individual
import operations as op

class TestIndividual(unittest.TestCase):
    def setUp(self):
        self.images = np.array( [[ 1, 2, 3],
                                 [ 1, 1, 1],
                                 [ 2, 0, 1]])
        self.eigenfaces = np.array( [[ 1, 1, 1],
                                     [ 4, 0, 6],
                                     [ 0, 3, 2]])

    def test_constructor_creates_individual_correctly(self):
        ind = Individual(self.images, self.eigenfaces, 2)

        self.assertTrue((ind.images_set == self.images).all())
        self.assertTrue((ind.eigenfaces == self.eigenfaces).all())
        self.assertEqual(ind.id, 2)