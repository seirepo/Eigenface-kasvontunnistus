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
        self.initialize()

    def load_data(self):
        """Load data from scikit learn datasets
        """
        abspath = os.path.abspath(__file__)
        path = Path(abspath)
        data_path = path.parents[1]/"data"

        faces = datasets.fetch_olivetti_faces(data_home=data_path)

        self.data = faces

    def create_individuals(self):
        """Create individual objects representing a class for one person
        """
        if len(self.individuals) == 0:
            images = self.data.images
            images_target = self.data.target
            for id in range(0,40):
                ims = op.images_to_vectors(images[np.where(images_target==id)])
                self.individuals.append(Individual(id, ims))

    def calculate_eigenfaces(self):
        """Collect a set of training images and calculate eigenfaces based on them
        """
        if self.eigenfaces is None:
            training_images = self.get_training_images()
            self.eigenfaces = op.calculate_eigenfaces(training_images)

    def project_faces(self) -> np.array:
        """Projects each individuals training images to the eigenface 
        spanned space and saves the coordinates in individual attribute

        Returns:
            np.array: an array of projected images as column vectors
        """
        projected_images = []
        for individual in self.individuals:
            projected = []
            training_images = individual.get_training_images()
            for image in training_images.T:
                proj_im = self.project_image(image)
                projected.append(proj_im)
                projected_images.append(proj_im)
            individual.set_image_coordinates(np.array(projected).T)
        return np.array(projected_images).T

    def project_image(self, im: np.array) -> np.array:
        """Project given image to eigenface space to get the coordinates

        Args:
            im (np.array): image to be projected

        Returns:
            np.array: coordinates in eigenface space
        """
        average_im = op.get_average_face(self.get_training_images())
        coordinates = op.get_coordinates(im, self.eigenfaces, average_im)
        return coordinates

    def classify_faces(self, k: int, p: int):
        """Classify test images of each individual and save the results to
        each individual

        Args:
            k (int): number of neighbors
            p (int): degree of the norm
        """
        for individual in self.individuals:
            test_ims = individual.get_test_images()
            res = []
            for im in test_ims.T:
                coordinates = self.project_image(im)
                distances = self.calculate_distances(coordinates, p)
                distances = sorted(distances, key=lambda i: i["dist"])
                nearest_k = distances[:k] # distances[0] keys: 'dist', 'id', 'coords'
                result = self.get_nearest(nearest_k)
                res.append({"test_im": im, "nearest_id": result["id"], "nearest_im_crds": result["coords"]})
            individual.set_nearest_neighbor(res)

    def get_nearest(self, nearest: list) -> dict:
        """Finish the classificaiton: finds the id and image
        coordinates of the most common class

        Args:
            nearest (list): data of the k nearest neighbors

        Returns:
            dict: with keys 'dist', 'id', 'coords'
        """
        nearest_ids = []
        for near in nearest:
            nearest_ids.append(near["id"])

        nearest_id = op.get_most_frequent(nearest_ids)

        class_min_dist = None
        for near in nearest:
            if near["id"] == nearest_id:
                dist = near["dist"]
                if class_min_dist is None:
                    class_min_dist = near
                elif dist < class_min_dist["dist"]:
                    class_min_dist = near
        return class_min_dist

    def calculate_distances(self, im: np.array, p: int) -> list:
        """Calculates distances between the given image coordinates and
        coordinates of the training images of each individual

        Args:
            im (np.array): an array of coordinates
            p (int): order of the norm

        Returns:
            list: list of dicts with key values 'dist', 'id', 'coords'
        """
        result = []
        for individual in self.individuals:
            images = individual.get_image_coordinates()
            id = individual.get_id()
            for image in images.T:
                distance = np.linalg.norm((image-im), p)
                result.append({"dist": distance, "id": id, "coords": image})
        return result

    def initialize(self):
        """Initialize the app
        """
        self.load_data()
        self.create_individuals()
        self.calculate_eigenfaces()
        self.project_faces()

    def classify(self, k=1, p=1):
        """Classifies the test images
        """
        self.classify_faces(k,p)
        print(f"k = {k}, p = {p}")
        self.print_results()

        av_face = op.get_average_face(self.get_training_images())
        #print("rekonstruoidaan jotkut training setin kasvot:")
        # vaikea tunnistaa: 0, 2, 3, 7, 8, 9, 22, 34, 39
        sel = self.individuals[9].get_training_images()[:,2]
        #sel = self.individuals[39].get_training_images()[:,2]
        #sel = self.individuals[22].get_test_images()[:,0]
        #print("eigenfaces: ", self.eigenfaces.shape)
        #plot.imshow(sel.reshape((64,64)), cmap="Greys_r")
        #plot.show()
        #coords = op.get_coordinates(sel, self.eigenfaces, av_face)
        #proj = op.get_projection(coords, self.eigenfaces, av_face)
        #proj2 = op.get_projection2(coords, self.eigenfaces, av_face)
        #plot.imshow(proj.reshape((64,64)), cmap="Greys_r")
        #plot.show()
        #plot.imshow(proj2.reshape((64,64)), cmap="Greys_r")
        #plot.show()
        #print("ero: ", sum(abs(proj - proj2) < 0.0001))

        #self.show_images(np.vstack([sel, proj, difference]).T)

        #self.show_images(self.eigenfaces[:,:15])

    def print_results(self):
        """Prints the result and id's with incorrect classification
        """
        count = len(self.get_test_images().T)
        corr = 0
        inc = 0
        wrong = []
        for ind in self.individuals:
            nearest = ind.get_nearest_neighbor()
            id = ind.get_id()
            for near in nearest:
                corr_tmp = 0
                inc_tmp = 0
                id_nearest = near["nearest_id"]
                if id == id_nearest:
                    corr_tmp += 1
                else:
                    wrong.append((id, id_nearest))
                    inc_tmp += 1
                corr += corr_tmp
                inc += inc_tmp
        print(f"oikein: {corr}, väärin: {inc}, kuvia yhteensä: {count}")
        print(f"tulos: {corr/count * 100} % oikein, {inc/count * 100} % väärin")
        print(f"Tunnistettu väärin:")
        for w in wrong:
            print(f"{w[0]}: {w[1]}")
        print(wrong)

    def get_projected_image(self, crds: np.array) -> np.array:
        """Return the image reconstructed from the given coordinates

        Args:
            crds (np.array): image coordinates as an array

        Returns:
            np.array: the resulting image
        """
        average = self.get_average_face()
        return op.get_projection(crds, self.eigenfaces, average)

    def get_average_face(self) -> np.array:
        """Return the average face calculated from the training images

        Returns:
            np.array: average face
        """
        return op.get_average_face(self.get_training_images())

    def get_training_images(self) -> np.array:
        """Get all training images from individual objects

        Returns:
            np.array: training images in an array as column vectors
        """
        training = []
        for individual in self.individuals:
            tr = individual.get_training_images()
            training.append(tr)
        training_array = np.hstack(training)
        return training_array

    def get_test_images(self) -> np.array:
        """Get all test images from individual objects

        Returns:
            np.array: test images in an array as column vectors
        """
        test = []
        for individual in self.individuals:
            ts = individual.get_test_images()
            test.append(ts)
        test_array = np.hstack(test)

        return test_array

    def get_individuals(self) -> list:
        return self.individuals

    def get_image_of_everyone(self):
        """Return a list of tuples containing id and first image of each individual

        Returns:
            list[tuple]: list of tuples
        """
        images = []
        for individual in self.individuals:
            im = individual.get_training_images()[:,0]
            id = individual.get_id()
            images.append((id, im))
        return images

    def show_images(self, images: np.array):
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

#app = App()
#app.classify()
#app.suorita()
