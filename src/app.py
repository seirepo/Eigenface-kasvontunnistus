from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plot, cm
from individual import Individual
import operations as op
from sklearn import datasets

class App:

    def __init__(self):
        self.individuals = []
        self.data = self.load_data()
        self.all_images = self.data.data.T
        self.eigenfaces = None
        self.training_images = None
        self.test_images = None

    def load_data(self):
        """load images

        Returns:
            sklearn.utils.Bunch: bunch that contains images, image array and target
        """
        abspath = os.path.abspath(__file__)
        path = Path(abspath)
        data_path = path.parents[1]/"data"

        faces = datasets.fetch_olivetti_faces(data_home=data_path)

        return faces

    def create_individuals(self):
        images = self.data.images
        images_target = self.data.target
        for id in range(0,40):
            ims = op.images_to_vectors(images[np.where(images_target==id)])
            self.individuals.append(Individual(id, ims))

    def alusta(self):
        self.create_individuals()

    def calculate(self):
        """Collect a set of training and test images, and calculate eigenfaces based on them
        """
        if self.eigenfaces is None:
            self.training_images, self.test_images = self.get_training_test_images()
            self.eigenfaces = op.calculate_eigenfaces(self.training_images, 320)

    def get_training_test_images(self):
        training = []
        test = []
        for individual in self.individuals:
            tr = individual.get_training_images()
            ts = individual.get_test_images()
            training.append(tr)
            test.append(ts)
        training_array = np.hstack(training)
        test_array = np.hstack(test)

        return training_array, test_array

    def get_all_images(self):
        return self.all_images

    def get_random_image(self):
        im = self.all_images[:,19].reshape((64,64))
        im = np.uint8(im*255)
        return im

    def get_image_of_everyone(self):
        images = []
        i = 0
        for individual in self.individuals:
            im = individual.get_training_images()[:,0].reshape((64,64))
            id = individual.get_id()
            images.append((id, im))
            i += 1
        return images

    def suorita(self):
        #self.create_individuals()
        self.calculate()

        # kuva joka projisoidaan
        test_im = self.training_images[:,0].reshape((4096,1))

        size, ims = self.eigenfaces.shape
        test_eigenfaces = self.eigenfaces

        test_im = self.training_images[:,74]

        test_im_coord = np.zeros((size,ims))

        for i in range(ims):
            mult = np.dot(test_im, test_eigenfaces[:,i])
            test_im_coord[:,i] = mult * test_eigenfaces[:,i]

        test_im_coord = np.sum(test_im_coord, axis=1)
        print(test_im_coord[:10])
        test2 = op.get_coordinates(test_im, self.eigenfaces)
        print(test2[:10])
        print(test_im[:10])

        plot.imshow(test_im.reshape((64,64)), cmap="Greys_r")
        plot.show()
        plot.imshow(test_im_coord.reshape((64,64)), cmap="Greys_r")
        plot.show()


    def show_images(self, images):
        """
        Function to plot images
        """
        im_count = images.shape[1]
        fig = plot.figure(figsize=(5,5))
        columns = 10
        rows = im_count//10+1
        for i in range(images.shape[1]):
            fig.add_subplot(rows, columns, i+1)
            im_vector = images[:,i]
            im_matrix = im_vector.reshape((64, 64))
            plot.imshow(im_matrix, cmap="Greys_r")#, vmin=0, vmax=1.0)
            plot.axis('off')

        fig.tight_layout()
        plot.show()

    def calculate_eigfaces_using_cov_mat(self, training_images, average_face):
        pass
        # calculate eigenfaces using covariance matrix
        #A = np.subtract(training_images, average_face)
        #S = np.cov(A)
        #vals, d = np.linalg.eig(S)
        #ind = vals.argsort()[::-1]
        #print(d.shape)
