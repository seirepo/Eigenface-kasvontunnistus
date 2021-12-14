import unittest
import numpy as np
from unittest.mock import Mock, patch
from app import App

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = Mock()
        self.rng = np.random.default_rng(seed=42)

    def test_get_training_images_returns_matrix_correct_shape(self):
        individual = Mock()
        individual.get_training_images.return_value = np.arange(0,8).reshape((-1,2))
        individual.get_test_images.return_value = np.arange(0,4).reshape((2,2))
        self.app.individuals = [individual, individual]
        train = App.get_training_images(self.app)
        test = App.get_test_images(self.app)

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

        train = App.get_training_images(self.app)
        test = App.get_test_images(self.app)


        self.assertTrue((train == train_should_be).all())
        self.assertTrue((test == test_should_be).all())

    def test_project_individuals_returns_correct_array(self):
        ind1 = Mock()
        ind2 = Mock()

        a1 = np.array([[1,2,3],[1,-1,1],[2,0,1]])
        a2 = np.array([[0,1,0],[1,2,1],[1,0,-2]])
        ind1.get_training_images.return_value = a1
        ind2.get_training_images.return_value = a2
        self.app.individuals = [ind1, ind2]
        self.app.eigenfaces = np.eye(3)
        self.app.project_image.side_effect = [ a1[:,0], a1[:,1], a1[:,2], a2[:,0], a2[:,1], a2[:,2] ]

        result = App.project_faces(self.app)
        should_be = np.hstack((a1, a2))

        self.assertEqual(result.shape, (3, 6))
        self.assertTrue((result == should_be).all())

    def test_calculate_distances_returns_list_of_correct_length(self):
        ind1 = Mock()
        ind2 = Mock()

        ind1.get_id.return_value = 1
        ind1.get_image_coordinates.return_value = self.rng.random((320, 4))
        ind2.get_id.return_value = 2
        ind2.get_image_coordinates.return_value = self.rng.random((320, 4))

        self.app.individuals = [ind1, ind2]
        im = self.rng.random((320))

        result = App.calculate_distances(self.app, im)
        self.assertEqual(len(result), 8)

    def test_calculate_distances_returns_correct_list(self):
        ind1 = Mock()
        ind2 = Mock()

        a1 = np.array([[1,2,3],[1,-1,1],[2,0,1]])
        a2 = np.array([[0,1,0],[1,2,1],[1,0,-2]])

        ind1.get_id.return_value = 1
        ind1.get_image_coordinates.return_value = a1
        ind2.get_id.return_value = 2
        ind2.get_image_coordinates.return_value = a2

        self.app.individuals = [ind1, ind2]
        im = np.arange(1,4)
        result = App.calculate_distances(self.app, im)
        should_be = [(2.0, 1), (7.0, 1), (5.0, 1), (4.0, 2), (3.0, 2), (7.0, 2)]

        self.assertEqual(result, should_be)

    def test_app_has_correct_attributes_before_any_operations(self):
        app = App()

        self.assertEqual(len(app.individuals), 0)
        self.assertIsNone(app.eigenfaces)

    def test_app_eigenfaces_are_not_calculated_more_than_once(self):
        pass

    def test_individuals_are_not_created_more_than_once(self):
        app = App()
        app.load_data()
        app.create_individuals()
        ind1 = app.individuals
        app.create_individuals()
        ind2 = app.individuals

        self.assertEqual(len(ind1), len(ind2))

    def test_app_has_correct_attributes_after_suorita(self):
        app = App()
        app.classify()

        self.assertEqual(len(app.individuals), 40)
        self.assertEqual(app.eigenfaces.shape[0], 4096)

    def test_get_image_of_everyone_returns_list_of_correct_length(self):
        app = App()
        app.load_data()
        app.create_individuals()
        count = len(app.individuals)
        images = app.get_image_of_everyone()

        self.assertEqual(len(images), count)

    def test_get_image_by_id_raises_exception_if_incorrect_id_is_given(self):
        app = App()
        app.load_data()
        app.create_individuals()

        self.assertRaises(ValueError, app.get_image_by_id, -4)
        self.assertRaises(ValueError, app.get_image_by_id, 9.5)
        self.assertRaises(ValueError, app.get_image_by_id, len(app.individuals) + 10)

    def test_calculate_knn_raises_exception_with_incorrect_image_input(self):
        app = App()
        app.load_data()
        app.create_individuals()
        app.calculate_eigenfaces()
        app.project_faces()

        im1 = self.rng.random((4097))
        im2 = self.rng.random((4096, 4096))
        im3 = self.rng.random((1, 2, 3))

        self.assertRaises(ValueError, app.calculate_knn, im1, 3)
        self.assertRaises(ValueError, app.calculate_knn, im2, 3)
        self.assertRaises(ValueError, app.calculate_knn, im3, 3)
