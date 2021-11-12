from matplotlib import pyplot as plot
import numpy as np

class Individual:
    def __init__(self, im_set_matrix):
        self.images_set = self.matrix_to_vectors(im_set_matrix)

    @staticmethod
    def matrix_to_vectors(im_set_matrix):
        return np.array([[0]])

    @staticmethod
    def matrix2d_to_column_vector(im_matrix):
        return np.array([[0]])

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