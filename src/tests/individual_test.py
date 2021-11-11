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
        
        #arr = np.array( [[ 1, 2, 3],
        #                [ 4, 2, 5]] )

    def test_matrix_to_vectors_returns_correct_number_of_vectors(self):
        test_matrix = np.array([ [self.m1, self.m2, self.m3] ])
        result = Individual.matrix_to_vectors(test_matrix)
        # should be "np.array( [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
        #                      [13, 14, 15, 16, 17, 18]] )"
        #self.assertEqual(result[0], np.array([1, 2, 3, 4, 5, 6]))
        self.assertEqual(result[0].tolist(), np.array([1, 2, 3, 4, 5, 6]).tolist())