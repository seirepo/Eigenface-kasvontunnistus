from matplotlib import pyplot as plot
import numpy as np

class Individual:
    """
    Class for one individual and their images as column vectors
    in matrix images_set
    """
    def __init__(self, im_set_matrix):
        """Constructor

        Args:
            im_set_matrix (np.array): np.array containing images
            of the person as np.arrays
        """
        self.images_set = Individual.matrix3d_submatrices_to_colums(im_set_matrix)

    @staticmethod
    def matrix3d_submatrices_to_colums(im_set_matrix):
        """
        Static function to turn n*n image np.arrays in an np.array
        to column vectors of length n*n

        Args:
            im_set_matrix (np.array): 3-dim np.array containing images as np.array

        Raises:
            ValueError: if arg is not a 3-dim np.array

        Returns:
            np.array: array containing images as column vectors
        """
        if im_set_matrix.ndim != 3:
            raise ValueError("invalid shape: " + str(im_set_matrix.shape))
        (count, row, col) = im_set_matrix.shape
        result = np.empty((row*col, count))

        for i in range(0, count):
            result[:,i] = im_set_matrix[i].flatten()

        return result

    @staticmethod
    def matrix2d_to_column_vector(im_matrix):
        """Static function to turn 2-dim np.arrays to a vector.
        Probably not necessary

        Args:
            im_matrix (np.array): 2-dim np.array

        Returns:
            np.array: given arg as a column vector
        """
        return np.array([im_matrix.flatten()]).T

    def show_images(self):
        """
        Function to plot images
        """
        fig = plot.figure(figsize=(5, 5))
        columns = 5
        rows = 2
        print(self.images_set[:,0])
        for i in range(self.images_set.shape[1]):
            fig.add_subplot(rows, columns, i+1)
            im_vector = self.images_set[:,i]
            im_matrix = im_vector.reshape((64, 64))
            plot.imshow(im_matrix, cmap="Greys_r")
        plot.show()

        #fig = plot.figure(figsize=(5, 5))
        #columns = 5
        #rows = 2
        #for i in range(1, columns*rows +1):
        #    fig.add_subplot(rows, columns, i)
        #    plot.imshow(self.im_matrix_array[i-1], cmap="Greys_r")
        #plot.show()
