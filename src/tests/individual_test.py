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
