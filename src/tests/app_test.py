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

        result = App.calculate_distances(self.app, im, 1)
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
        result = App.calculate_distances(self.app, im, 1)
        for r in result:
            r["coords"] = str(r["coords"])
        should_be = [
            {"dist": 2.0, "id": 1, "coords": "[1 1 2]"},
            {"dist": 7.0, "id": 1, "coords": "[ 2 -1  0]"},
            {"dist": 5.0, "id": 1, "coords": "[3 1 1]"},
            {"dist": 4.0, "id": 2, "coords": "[0 1 1]"},
            {"dist": 3.0, "id": 2, "coords": "[1 2 0]"},
            {"dist": 7.0, "id": 2, "coords": "[ 0  1 -2]"},
        ]
        self.assertEqual(result, should_be)

    def test_app_has_correct_attributes_set_before_any_operations(self):
        app = App()
        self.assertFalse(len(app.individuals) == 0)
        self.assertIsNotNone(app.eigenfaces)

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

    def test_get_image_by_id_returns_correct_image(self):
        im1 = self.rng.random((4000, 3))
        im2 = self.rng.random((4000, 3))
        im3 = self.rng.random((4000, 3))
        ind1 = Mock()
        ind2 = Mock()
        ind3 = Mock()
        ind1.get_training_images.return_value = im1
        ind2.get_training_images.return_value = im2
        ind3.get_training_images.return_value = im3
        ind1.get_id.return_value = 1
        ind2.get_id.return_value = 2
        ind3.get_id.return_value = 3
        self.app.individuals = [ind1, ind2, ind3]
        self.assertTrue((App.get_image_by_id(self.app, 1) == im1[:,0]).all())
        self.assertTrue((App.get_image_by_id(self.app, 2) == im2[:,0]).all())
        self.assertTrue((App.get_image_by_id(self.app, 3) == im3[:,0]).all())

    def test_get_nearest_returns_correct_dict(self):
        lst = [{"id": 1, "dist": 3}, {"id": 1, "dist": 2}, {"id": 2, "dist": 0}]
        result = App.get_nearest(self.app, lst)
        self.assertEqual(result, lst[1])

        lst = [{"id": 2, "dist": 3}, {"id": 1, "dist": 2}, {"id": 2, "dist": 0}]
        result = App.get_nearest(self.app, lst)
        self.assertEqual(result, lst[2])

        lst = [{"id": 2, "dist": 3}, {"id": 1, "dist": 2}, {"id": 3, "dist": 0}]
        result = App.get_nearest(self.app, lst)
        self.assertEqual(result, lst[0])
