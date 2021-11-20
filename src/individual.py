import numpy as np
import operations as op

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
        self.training_set = np.array([]) # len = 0 jos ei alustettu
        self.eigenfaces = np.array([]) # len = 0
        self.id = id

    def get_images(self):
        return self.images

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
