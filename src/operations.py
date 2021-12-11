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
    given image set, forming an orthonormal base.
    If k is not given, then the number of eigenfaces needed to represent
    97.5 % of the total variance is returned

    Args:
        training_images (np.array): training set images
        k (int): number of eigenfaces to return

    Returns:
        np.array: k eigenfaces in an array
    """

    im_count = training_images.shape[1]

    if k > im_count:
        raise Exception(f"Cannot return more eigenfaces than images: {k} > {im_count}")

    #average_face = np.mean(training_images, axis=1).reshape((-1, 1))
    average_face = get_average_face(training_images)
    difference_faces = np.subtract(training_images, average_face)

    ATA = np.matmul(difference_faces.T, difference_faces)
    vals, eig_vectors = np.linalg.eig(ATA)

    #if not (vals > 0).all():
    #    raise Exception(f"A^TA of the given matrix \n {training_images} \n has eigenvalues <= 0: {vals}")

    indx = vals.argsort()[::-1]

    if k <= 0:
        eigvals = sorted(vals.tolist())[::-1]
        count = len(eigvals)
        eigsum = sum(eigvals)
        csum = 0
        for i in range(0,count):
            csum = csum + eigvals[i]
            total_variance = csum / eigsum
            if total_variance > 0.85:
                k = i
                break

    selected = eig_vectors[:,indx][:,:k]

    eigenfaces = np.matmul(difference_faces, selected)

    #scaled_eigenfaces = np.interp(eigenfaces, (eigenfaces.min(), eigenfaces.max()), (0, 1))
    #try:
    #    scaled = np.interp(selected, (selected.min(), selected.max()), (0, 1))
    #except ValueError:
    #        print(f"skaalaus ei onnistunut, skaalattava matriisi {selected} on tyhjä")
    #result = np.linalg.qr(scaled_eigenfaces)[0]
    result = np.linalg.qr(eigenfaces)[0]

    return result

def euclidean_distance2(im1, im2):
    if im1.shape[0] != im2.shape[0]:
        raise ValueError(f"Illegal size of input vectors {im1.shape} and {im2.shape}: {im1.shape[0]} != {im2.shape[0]}")
    return np.sum((im1 - im2)**2)

def pnorm(im1, im2, p):
    return np.linalg.norm((im1-im2), p)

def get_most_frequent(values):
    vals, counts = np.unique(values, return_counts=True)
    result = zip(counts, vals)
    result = list(result)
    sorted_res = sorted(result)[::-1]
    max_count = sorted_res[0][0]
    max_pair = sorted_res[0]

    tie = []
    for item in sorted_res[1:]:
        if item[0] == max_count:
            tie.append(item)

    if len(tie) == 0:
        return max_pair[1]
    else:
        max_index = values.index(max_pair[1])
        for item in tie:
            if values.index(item[1]) < max_index:
                max_pair = item
    return max_pair[1]

def get_average_face(training_images):
    return np.mean(training_images, axis=1).reshape((-1,1))

def get_coordinates(image, basis):
    """Calculates coordinates of a given image in the given face space.
    The basis must be orthonormal

    Args:
        image (np.array): image
        space (np.array): eigenface vectors spanning a face space

    Returns:
        np.array: coordinates
    """
    ims = basis.shape[1]
    coordinates_list = []
    for i in range(ims):
        mult = np.dot(image, basis[:,i])
        coordinates_list.append(mult)
    coordinates = np.array(coordinates_list)
    return coordinates

def get_projection(image, basis):
    """Returns the projection of an image on the given face space.
    The basis of the space must be orthonormal

    Args:
        image (np.array): image
        basis (np.array): eigenface vectors spanning a face space

    Returns:
        np.array: projection of the image
    """
    size, ims = basis.shape
    weights = np.zeros((size, ims))
    for i in range(ims):
        mult = np.dot(image, basis[:,i])
        weights[:,i] = mult * basis[:,i]
    weights = np.sum(weights, axis=1)
    return weights
