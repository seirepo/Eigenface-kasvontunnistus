import unittest
import numpy as np
from unittest.mock import Mock, patch
from app import App

class TestApp(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_training_images_returns_matrix_correct_shape(self):
        individual = Mock()
        individual.get_training_images.return_value = np.arange(0,8).reshape((-1,2))
        individual.get_test_images.return_value = np.arange(0,4).reshape((2,2))
        #with patch.object(App, 'load_data') as mock_load:
        #    mock_load.return_value = [1, 2, 3]
        #    app = App()
        #    print(app.data)
        #    assert 0
            
        app = Mock()
        app.individuals = [individual, individual]
        training, test = App.get_training_test_images(app)

        self.assertEqual(training.shape, (4, 4))
        self.assertEqual(test.shape, (2, 4))

