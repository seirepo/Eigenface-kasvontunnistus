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
        self.training_images, self.test_images = self.train_test_split() #np.array([]) # len = 0 jos ei alustettu
        self.eigenfaces = np.array([]) # len = 0
        self.id = id

    def get_images(self):
        return self.images

    def train_test_split(self):
        return (self.images[:,:8], self.images[:,8:])

    def get_training_images(self):
        return self.training_images

    def get_test_images(self):
        return self.test_images

    # for testing only
    def calculate_eigenfaces(self):
        average_face = np.mean(self.images_set, axis=1).reshape((-1, 1))
        faces_minus_average = np.subtract(self.images_set, average_face)

        L = np.matmul(faces_minus_average.T, faces_minus_average)
        v = np.linalg.eig(L)[1]
        eigenfaces = np.zeros((4096, 10))
        M = 10
        for i in range(0, M):
            for j in range(0, M):
                eigenfaces[:,i] += v[i][j] * faces_minus_average[:,j]

        return eigenfaces
