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
        self.projected_images = None

    def get_images(self):
        return self.images

    def train_test_split(self):
        return (self.images[:,:8], self.images[:,8:])

    def get_training_images(self):
        return self.training_images

    def get_test_images(self):
        return self.test_images

    def set_projected_images(self, images):
        self.projected_images = images
