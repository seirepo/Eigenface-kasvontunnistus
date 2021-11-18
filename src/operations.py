import numpy as np

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

def calculate_eigenfaces(training_images, k=-1):
    """Calculates and returns k eigenfaces with the largest eigenvalue of a
    given image set. If k is not given, every eigenface is returned

    Args:
        training_images (np.array): training set images
        k (int): number of eigenfaces to return

    Returns:
        np.array: k eigenfaces in an array
    """

    im_count = training_images.shape[1]

    if k > im_count - 1:
        raise Exception(f"Cannot return more eigenfaces than images: {k} > {im_count}")

    if k < 0:
        k = im_count

    # laske kuvien keskiarvo ja vähennä se niistä
    average_face = np.mean(training_images, axis=1).reshape((-1, 1))

    faces_minus_average = np.subtract(training_images, average_face)

    # laske apumatriisi ja sen ominaisvektorit
    ATA = np.matmul(faces_minus_average.T, faces_minus_average)
    vals, eig_vectors = np.linalg.eig(ATA)

    # järjestä ominaisarvot laskevaan järjestykseen
    indx = vals.argsort()[::-1]

    # laske apumatriisin ominaisvektorien avulla kuvamatriisin ominaisvektorit
    eigenfaces = np.zeros((4096, im_count))

    for i in range(0, im_count):
        for j in range(0, im_count):
            eigenfaces[:,i] += eig_vectors[i][j] * faces_minus_average[:,j]

    # valitse lasketuista ominaiskasvoista k suurinta ominaisarvoa vastaavat
    return eigenfaces[:,indx][:,:k]

def get_images_with_mean_subtracted(images):
    pass
