import numpy as np

def sum_matrices(a, b):
    return a + b

def matrix3d_submatrices_to_columns(im_set_matrix):
    """
    If the image set is given as a matrix containing k images as nxm matrices,
    this function returns a 2-dimensional (nxm)xk-matrix containing the images
    flattened as column vectors

    Args:
        im_set_matrix (np.array): 3-dim np.array containing images as np.arrays

    Raises:
        ValueError: if given arg is not a 3-dim np.array

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

def calculate_eigenfaces(training_images, k):
    """Calculates and returns k eigenfaces with the largest eigenvalue of a
    given image set

    Args:
        training_images (np.array): training set images
        k (int): number of eigenfaces to return

    Returns:
        np.array: k eigenfaces in an array
    """
    if k > training_images.shape[1] - 1:
        raise Exception(f"Cannot return more eigenfaces than images: {k} > {training_images.shape[1]}")

    # laske kuvien keskiarvo ja vähennä se niistä
    average_face = np.mean(training_images, axis=1).reshape((-1, 1))
    #print("keskiarvo: ", self.average_face, self.average_face.shape)
    #Individual.show_images(self.average_face)

    faces_minus_average = np.subtract(training_images, average_face)
    #print("kasvot joista vähennetty keskiarvo: ", faces_minus_average.shape)
    #Individual.show_images(faces_minus_average)

    # laske apumatriisi ja sen ominaisvektorit
    L = np.matmul(faces_minus_average.T, faces_minus_average)
    vals, eig_vectors = np.linalg.eig(L)

    # joku tarkistus sille että k on järkevä (0 < k < len(vects))
    #if k > len(vals) - 1:
    #    raise Exception(f"calculate_eigenfaces: illegal argument {k}")

    # järjestä ominaisarvot laskevaan järjestykseen
    indx = vals.argsort()[::-1]

    # laske apumatriisin ominaisvektorien avulla kuvamatriisin ominaisvektorit
    eigenfaces = np.zeros((4096, 10))
    M = 10
    for i in range(0, M):
        for j in range(0, M):
            eigenfaces[:,i] += eig_vectors[i][j] * faces_minus_average[:,j]

    # valitse lasketuista ominaiskasvoista k suurinta ominaisarvoa vastaavat
    return eigenfaces[:,indx][:,:k]

def get_images_with_mean_subtracted(images):
    pass
