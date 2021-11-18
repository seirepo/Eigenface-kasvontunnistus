import numpy as np

class Individual:
    """
    Class for one individual and their images as column vectors
    in matrix images_set
    """
    def __init__(self, images, eigenfaces=None, id=None):
        """Constructor

        Args:
            im_set_matrix (np.array): np.array containing images
            of the person as np.arrays
        """
        self.images_set = images
        self.eigenfaces = eigenfaces
        self.id = id

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
