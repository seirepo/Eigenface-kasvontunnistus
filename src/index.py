from pathlib import Path
import os
import random
import numpy as np
from matplotlib import pyplot as plot
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
    average_face = op.get_average_face(training_images)
    eigenfaces = op.calculate_eigenfaces(training_images)


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
        plot.imshow(im_matrix, cmap="Greys_r")
        plot.axis('off')

    fig.tight_layout()
    plot.show()

    #fig = plot.figure(figsize=(5, 5))
    #columns = 5
    #rows = 2
    #for i in range(1, columns*rows +1):
    #    fig.add_subplot(rows, columns, i)
    #    plot.imshow(self.im_matrix_array[i-1], cmap="Greys_r")
    #plot.show()

def test1(images):
    random.seed(10)

    # tallennetaan data individuals-listaan, toistaiseksi vain henkilö jonka
    # target = 0
    #individuals = []
    #for i in range(0,40):
        #individuals.append(images[np.where(images_target==0)])

    # tallennetaan ensimmäisen henkilöt kuvat ja lasketaan niistä eigenfacet
    #images = op.images_to_vectors(individuals[0])

    #valitaan kuvista 80 % training settiin ja lasketaan niille eigenfacet
    idx = range(len(images.T))
    indices = random.sample(idx, 8)
    training_images = images[:,indices]
    #print("valitut indeksit: ", indices)
    im_count = len(training_images.T)
    #print("kuvia: ", im_count)
    training1 = training_images[:,:im_count//2]
    training2 = training_images[:,im_count//2:]
    
    # alkeellinen testi että onnistui
    print(training1.shape, training2.shape)
    print((training1 == training_images[:,:4]).all())
    print((training2 == training_images[:,4:]).all())

    eigenfaces1 = op.calculate_eigenfaces(training1, 2)
    mean1 = op.get_average_face(training1)
    eigenfaces2 = op.calculate_eigenfaces(training2, 2)
    mean2 = op.get_average_face(training2)

    # projisoidaan joku training setin kuva noille
    sel = training2[:,2].reshape((-1,1))
    eig1 = eigenfaces2[:,0]
    eig2 = eigenfaces2[:,1]
    print("eigit: ", eig1.shape, eig2.shape)
    print("valittu & mean: ", sel.shape, mean2.shape)
    diff = np.subtract(sel, mean2)
    print("erotus: ", diff.shape, diff[:5])
    a = np.matmul(eig1, diff)
    b = np.matmul(eig2, diff)
    print(a, b)
    coeffs = np.array([a,b])
    res = a*eig1 + b*eig2
    res = res.reshape((-1,1))
    print(res.shape)
    #test = np.array([a1, b1]).T
    #print(test[:5,:], a1[:5], b1[:5])
    final = mean2 + res
    print(f"{final[3]} = {mean2[3]} + {res[3]}")
    #print("dot product: ", np.dot(eig1, eig2))

    # näytetään tulos
    #show_images(eigenfaces1)
    #show_images(np.concatenate((eigenfaces1, training1), axis=1))
    show_images(eigenfaces2)
    show_images(np.concatenate((eigenfaces2, training2, res, final), axis=1))

if __name__ == "__main__":
    main()