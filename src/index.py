from pathlib import Path
import os
import random
import numpy as np
from matplotlib import pyplot as plot, scale
from individual import Individual
import operations as op

def main():
    # ladataan aineisto
    abspath = os.path.abspath(__file__)
    path = Path(abspath)
    #print(list(path.parents))
    im_path = path.parents[1]/"data"/"olivetti_faces"/"olivetti_faces.npy"
    target_path = im_path.parents[0]/"olivetti_faces_target.npy"
    #print(target_path)

    images_target = np.load(target_path)
    images = np.load(im_path)

    # tallennetaan data individuals-listaan individual-olioina
    individuals = []
    for id in range(0,40):
        ims = op.images_to_vectors(images[np.where(images_target==id)])
        individuals.append(Individual(id, ims))

    # kerätään kaikkien kuvat yhteen matriisiin niin ne voi printtaa jos haluaa
    all_images = individuals[0].get_images()
    for i in range(1,len(individuals)):
        all_images = np.concatenate([all_images, individuals[i].get_images()], axis=1)

    # kerätään training- ja testisetti joissa on kaikkien training- ja testikuvat
    training_images, test_images = op.get_all_training_and_test_images(individuals)

    # laskeaan training-setistä keskiarvo ja eigenfacet
    eigenfaces = op.calculate_eigenfaces(training_images, 320)

    # kuva joka projisoidaan
    test_im = training_images[:,0].reshape((4096,1))

    size, ims = eigenfaces.shape
    test_eigenfaces = eigenfaces

    test_im = training_images[:,74]

    test_im_coord = np.zeros((size,ims))

    for i in range(ims):
        mult = np.dot(test_im, test_eigenfaces[:,i])
        test_im_coord[:,i] = mult * test_eigenfaces[:,i]

    test_im_coord = np.sum(test_im_coord, axis=1)
    print(test_im_coord[:10])
    test2 = op.get_coordinates(test_im, eigenfaces)
    print(test2[:10])
    print(test_im[:10])

    plot.imshow(test_im.reshape((64,64)), cmap="Greys_r")
    plot.show()
    plot.imshow(test_im_coord.reshape((64,64)), cmap="Greys_r")
    plot.show()


def show_images(images):
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

def calculate_eigfaces_using_cov_mat(training_images, average_face):
    pass
    # calculate eigenfaces using covariance matrix
    #A = np.subtract(training_images, average_face)
    #S = np.cov(A)
    #vals, d = np.linalg.eig(S)
    #ind = vals.argsort()[::-1]
    #print(d.shape)

if __name__ == "__main__":
    main()
