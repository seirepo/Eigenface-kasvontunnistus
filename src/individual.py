from matplotlib import pyplot as plot
import numpy as np
import operations as op

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
        self.images_set = op.matrix3d_submatrices_to_colums(im_set_matrix)
        self.training_set = None
        self.test_set = None
        self.average_face = None

    def calculate_eigenfaces(self):
        # valitse kuvat training settiin
        self.training_set = self.images_set[:,:8]
        self.test_set = self.images_set[:,8:10]

        # laske kuvien keskiarvo ja vähennä se niistä
        self.average_face = np.mean(self.training_set, axis=1).reshape((-1, 1))
        #print("keskiarvo: ", self.average_face, self.average_face.shape)
        #Individual.show_images(self.average_face)

        faces_minus_average = np.subtract(self.training_set, self.average_face)
        #print("kasvot joista vähennetty keskiarvo: ", faces_minus_average.shape)
        #Individual.show_images(faces_minus_average)

        # laske apumatriisi ja sen ominaisvektorit
        L = np.matmul(faces_minus_average.T, faces_minus_average)
        v = np.linalg.eig(L)[1]

        # laske apumatriisin ominaisvektorien avulla kuvamatriisin ominaisvektorit
        eigenfaces = np.zeros((4096, 8))
        M = 8
        for i in range(0, M):
            for j in range(0, M):
                eigenfaces[:,i] += v[i][j] * faces_minus_average[:,j]

        self.show_images(eigenfaces)

    @staticmethod
    def show_images(images):
        """
        Function to plot images
        """
        #(rows, columns) = images.shape
        fig = plot.figure(figsize=(5, 5))
        columns = 5
        rows = 2
        print(images[:,0])
        for i in range(images.shape[1]):
            fig.add_subplot(rows, columns, i+1)
            im_vector = images[:,i]
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
