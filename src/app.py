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
        self.data = None
        self.eigenfaces = None

    def load_data(self):
        """load images

        Returns:
            sklearn.utils.Bunch: bunch that contains images, image array and target
        """
        abspath = os.path.abspath(__file__)
        path = Path(abspath)
        data_path = path.parents[1]/"data"

        faces = datasets.fetch_olivetti_faces(data_home=data_path)

        self.data = faces

    def create_individuals(self):
        if len(self.individuals) == 0:
            images = self.data.images
            images_target = self.data.target
            for id in range(0,40):
                ims = op.images_to_vectors(images[np.where(images_target==id)])
                self.individuals.append(Individual(id, ims))

    def calculate_eigenfaces(self):
        """Collect a set of training and test images, and calculate eigenfaces based on them
        """
        if self.eigenfaces is None:
            training_images = self.get_training_images()
            self.eigenfaces = op.calculate_eigenfaces(training_images, 320)

    def get_training_images(self):
        training = []
        for individual in self.individuals:
            tr = individual.get_training_images()
            training.append(tr)
        training_array = np.hstack(training)
        return training_array

    def get_test_images(self):
        test = []
        for individual in self.individuals:
            ts = individual.get_test_images()
            test.append(ts)
        test_array = np.hstack(test)

        return test_array

    def get_individuals(self):
        return self.individuals

    def get_random_image(self):
        im = self.all_images[:,19].reshape((64,64))
        im = np.uint8(im*255)
        return im

    def get_image_of_everyone(self):
        images = []
        i = 0
        for individual in self.individuals:
            im = individual.get_training_images()[:,0]#.reshape((64,64))
            id = individual.get_id()
            images.append((id, im))
            i += 1
        return images

    def project_faces(self):
        projected_images = []
        for individual in self.individuals:
            projected = []
            average_images = individual.get_average_images()
            for image in average_images.T:
                proj_im = self.project_image(image)
                projected.append(proj_im)
                projected_images.append(proj_im)
            individual.set_image_coordinates(np.array(projected).T)
        return np.array(projected_images).T

    def calculate_knn(self, im, k):
        shape = im.shape
        if len(shape) == 1 and shape[0] != 4096:
            raise ValueError(f"Invalid input image shape {im.shape}")
        elif len(shape) == 2 and shape != (4096, 1):
            raise ValueError(f"Invalid input image shape {im.shape}")
        elif len(shape) > 2:
            raise ValueError(f"Invalid input image shape {im.shape}")

        coordinates = self.project_image(im)
        distances = self.calculate_distances(coordinates)
        distances.sort()
        if k >= len(distances):
            k = len(distances) - 1
        nearest = distances[:k]
        near = []
        for n in nearest:
            near.append(n[1])
        res = ", ".join([str(int) for int in near])
        return res

    def calculate_distances(self, im):
        distances = []
        for individual in self.individuals:
            images = individual.get_image_coordinates()
            id = individual.get_id()
            for image in images.T:
                distance = op.euclidean_distance2(image, im)
                distances.append((distance, id))
        return distances

    def suorita(self):
        self.load_data()
        self.create_individuals()
        self.calculate_eigenfaces()
        self.project_faces()
        #self.classify_faces()
        self.calculate_knn
        self.calculate_eigenfaces()

        print("ajetaan tunnistusalgoritmi kaikille testikuville")
        k = 5
        for individual in self.individuals:
            test_ims = individual.get_test_images()
            id = individual.get_id()
            print(f"id: {id}, lähimmät {k}")
            for im in test_ims.T:
                print(f"\t {self.calculate_knn(im, k)}")

    def project_image(self, im):
        """Project given image to eigenface space

        Args:
            im (np.array): image to be projected

        Returns:
            np.array: coordinates in eigenface space
        """
        coordinates = op.get_coordinates(im, self.eigenfaces)
        return coordinates








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

#app = App()
#app.alusta()
#app.suorita()