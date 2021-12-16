import operations as op
import numpy as np

class Individual:
    """
    Class for one individual and their images as column vectors
    in matrix images_set
    """
    def __init__(self, id, images):
        """Constructor

        Args:
            im_set_matrix (np.array): np.array containing images
            of the person as np.arrays
        """
        self.images = images
        self.training_images, self.test_images = self.train_test_split()
        self.id = id
        self.image_coordinates = None
        self.nearest_neighbor = dict()

    def get_images(self):
        return self.images

    def get_id(self):
        return self.id

    def train_test_split(self):
        return (self.images[:,:8], self.images[:,8:])

    def get_training_images(self):
        return self.training_images

    def get_test_images(self):
        return self.test_images

    def set_image_coordinates(self, images):
        self.image_coordinates = images

    def get_image_coordinates(self):
        return self.image_coordinates

    def set_nearest_neighbor(self, nearest: list):
        self.nearest_neighbor = nearest

    def get_nearest_neighbor(self):
        return self.nearest_neighbor
