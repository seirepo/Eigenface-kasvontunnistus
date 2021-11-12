from matplotlib import pyplot as plot
import numpy as np

class Individual:
    def __init__(self, im_set_matrix):
        self.images_set = Individual.matrix3d_submatrices_to_colums(im_set_matrix)

    @staticmethod
    def matrix3d_submatrices_to_colums(im_set_matrix):
        if im_set_matrix.ndim != 3:
            raise ValueError("invalid shape: " + str(im_set_matrix.shape))
        (n, r, c) = im_set_matrix.shape
        result = np.empty((r*c, n))

        for i in range(0, n):
            result[:,i] = im_set_matrix[i].flatten()

        return result

    @staticmethod
    def matrix2d_to_column_vector(im_matrix):
        return np.array([im_matrix.flatten()]).T

    """
    def show(self):
        fig = plot.figure(figsize=(5, 5))
        columns = 5
        rows = 2
        for i in range(1, columns*rows +1):
            fig.add_subplot(rows, columns, i)
            plot.imshow(self._im_matrix_array[i-1], cmap="Greys_r")
        plot.show()
        """