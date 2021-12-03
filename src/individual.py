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
        self.average_images = self.select_average()
        self.projected_images = None

    def get_images(self):
        return self.images

    def get_id(self):
        return self.id

    def train_test_split(self):
        return (self.images[:,:8], self.images[:,8:])

    def select_average(self):
        if self.training_images.shape[1] < 2:
            return np.array(self.training_images)

        count = self.training_images.shape[1]
        averages_list = []

        for i in range(0, count, 2):
            average = op.get_average_face(self.training_images[:,i:i+2]).flatten()
            averages_list.append(average)
        averages = np.array(averages_list).T
        return averages

    def get_training_images(self):
        return self.training_images

    def get_test_images(self):
        return self.test_images

    def get_average_images(self):
        return self.average_images

    def set_projected_images(self, images):
        self.projected_images = images
