from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plot
from individual import Individual
import operations as op

def main():
    abspath = os.path.abspath(__file__)
    path = Path(abspath)
    #print(list(path.parents))
    im_path = path.parents[1]/"data"/"olivetti_faces"/"olivetti_faces.npy"
    target_path = im_path.parents[0]/"olivetti_faces_target.npy"
    #print(target_path)

    images_target = np.load(target_path)
    images = np.load(im_path)
    #plot.imshow(images[1], cmap="Greys_r")
    #plot.show()

    individuals = []
    for i in range(0,40):
        individuals.append(images[np.where(images_target==0)])

    #print(individuals[0].shape)
    #test_individual = Individual(individuals[0])
    #print(type(individuals[0])) # numpy.ndarray
    #print("testihenkilön image set: ", test_individual.images_set.shape)
    #Individual.show_images(test_individual.images_set)

    #print("testi ja training set")
    #test_individual.calculate_eigenfaces()
    #print(test_individual.images_set.shape)
    #Individual.show_images(test_individual.training_set)
    #Individual.show_images(test_individual.test_set)
    #test_individual.calculate_eigenfaces()

    images = op.images_to_vectors(individuals[0])
    eigenfaces = op.calculate_eigenfaces(images)

    show_images(images)
    print(eigenfaces.shape)
    show_images(eigenfaces)

    test = Individual(images)
    asd = test.calculate_eigenfaces()

    show_images(asd)

    test = np.array([[1, 1], [2, 0]])
    res = op.calculate_eigenfaces(test, 2)
    print(res)

def show_images(images):
    """
    Function to plot images
    """
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


if __name__ == "__main__":
    main()