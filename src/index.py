import numpy as np
from matplotlib import pyplot as plot
from pathlib import Path
import os
from individual import Individual

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


    #fig = plot.figure(figsize=(5, 5))
    #columns = 5
    #rows = 2
    #for i in range(1, columns*rows +1):
    #    fig.add_subplot(rows, columns, i)
    #    plot.imshow(individuals[0][i-1], cmap="Greys_r")
    #plot.show()

    test_individual = Individual(individuals[0])
    print(type(individuals[0])) # numpy.ndarray
    #test_individual.show()



if __name__ == "__main__":
    main()