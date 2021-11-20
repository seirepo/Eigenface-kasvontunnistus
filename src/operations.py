import numpy as np

def images_to_vectors(im_set_matrix):
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
    given image set. If k is not given, then the number of eigenfaces
    needed to represent 80 % of the total variance is returned

    Args:
        training_images (np.array): training set images
        k (int): number of eigenfaces to return

    Returns:
        np.array: k eigenfaces in an array
    """

    im_len, im_count = training_images.shape

    if k > im_count:
        raise Exception(f"Cannot return more eigenfaces than images: {k} > {im_count}")


    # laske kuvien keskiarvo ja vähennä se niistä
    #average_face = np.mean(training_images, axis=1).reshape((-1, 1))
    average_face = get_average_face(training_images)

    faces_minus_average = np.subtract(training_images, average_face)

    # laske apumatriisi ja sen ominaisvektorit
    ATA = np.matmul(faces_minus_average.T, faces_minus_average)
    vals, eig_vectors = np.linalg.eig(ATA)

    # järjestä ominaisarvot laskevaan järjestykseen
    indx = vals.argsort()[::-1]

    # laske apumatriisin ominaisvektorien avulla kuvamatriisin ominaisvektorit
    eigenfaces = np.zeros((im_len, im_count))

    for i in range(0, im_count):
        for j in range(0, im_count):
            eigenfaces[:,i] += eig_vectors[i][j] * faces_minus_average[:,j]

    if k <= 0:
        eigvals = sorted(vals.tolist())[::-1]
        count = len(eigvals)
        eigsum = sum(eigvals)
        csum = 0
        for i in range(0,count):
            csum = csum + eigvals[i]
            total_variance = csum / eigsum
            if total_variance > 0.80:
                k = i
                break

    # valitse lasketuista ominaiskasvoista k suurinta ominaisarvoa vastaavat
    return eigenfaces[:,indx][:,:k]

def get_average_face(training_images):
    return np.mean(training_images, axis=1).reshape((-1,1))