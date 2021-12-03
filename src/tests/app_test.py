import unittest
import numpy as np
from unittest.mock import Mock, patch
from app import App

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = Mock()

    def test_get_training_images_returns_matrix_correct_shape(self):
        #with patch.object(App, 'load_data') as mock_load:
        #    mock_load.return_value = [1, 2, 3]
        #    app = App()
        #    print(app.data)
        #    assert 0

        individual = Mock()
        individual.get_training_images.return_value = np.arange(0,8).reshape((-1,2))
        individual.get_test_images.return_value = np.arange(0,4).reshape((2,2))
        self.app.individuals = [individual, individual]
        train, test = App.get_training_test_images(self.app)

        self.assertEqual(train.shape, (4, 4))
        self.assertEqual(test.shape, (2, 4))

    def test_get_training_images_returns_correct_arrays(self):
        ind1 = Mock()
        ind2 = Mock()

        a1 = np.arange(0,12).reshape((3,4))
        a2 = np.arange(12,24).reshape((3,4))
        b1 = np.arange(0,8).reshape((2,4))
        b2 = np.arange(8,16).reshape((2,4))

        ind1.get_training_images.return_value = a1
        ind1.get_test_images.return_value = b1
        ind2.get_training_images.return_value = a2
        ind2.get_test_images.return_value = b2

        self.app.individuals = [ind1, ind2]
        train_should_be = np.hstack((a1, a2))
        test_should_be = np.hstack((b1, b2))

        train, test = App.get_training_test_images(self.app)

        self.assertTrue((train == train_should_be).all())
        self.assertTrue((test == test_should_be).all())

    def test_project_individuals_returns_correct_array(self):
        ind1 = Mock()
        ind2 = Mock()

        a1 = np.array([[1,2,3],[1,-1,1],[2,0,1]])
        a2 = np.array([[0,1,0],[1,2,1],[1,0,-2]])
        ind1.get_average_images.return_value = a1
        ind2.get_average_images.return_value = a2
        self.app.individuals = [ind1, ind2]
        self.app.eigenfaces = np.eye(3)
        self.app.project_image.side_effect = [ a1[:,0], a1[:,1], a1[:,2], a2[:,0], a2[:,1], a2[:,2] ]

        result = App.project_individuals(self.app)
        should_be = np.hstack((a1, a2))

        self.assertEqual(result.shape, (3, 6))
        self.assertTrue((result == should_be).all())
