from matplotlib import image
import numpy as np

def images_to_vectors(im_set_matrix: np.array) -> np.array:
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

def calculate_eigenfaces(training_images: np.array, k=-1) -> np.array:
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
    average_face = get_average_face(training_images).reshape((-1, 1))
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
            if total_variance > 0.95:
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

def pnorm(im1: np.array, im2: np.array, p: int):
    return np.linalg.norm((im1-im2), p)

def get_most_frequent(values: list) -> int:
    """Returns the most frequent value from the given list. If no value
    is more frequent than te others, the first value from the original list
    is returned.

    Args:
        values (list): list of values

    Returns:
        int: the most frequent value
    """
    #print("\n------------------\n")
    #print("alkuperäinen lista: ", values)
    #ids = []
    #for v in values:
    #    ids.append(v["id"])
    #print("id: ", ids)
    ids = values
    #vals, counts = np.unique(values, return_counts=True)
    vals, counts = np.unique(ids, return_counts=True)
    #print(vals, counts)
    result = zip(counts, vals)
    result = list(result)
    #print(result)
    #sorted_res = sorted(result)[::-1]
    sorted_res = sorted(result, key=lambda d: d[0], reverse=True)
    #print(sorted_res)
    max_count = sorted_res[0][0]
    max_pair = sorted_res[0]

    tie = []
    for item in sorted_res[1:]:
        #print("tasa: ", item[0], max_count)
        if item[0] == max_count:
            #print(f"{item} lisätty tasapeliin")
            tie.append(item)

    #print("tasa ", tie)
    if len(tie) == 0:
        #print("palautetaan ", max_pair[1])
        return max_pair[1]
    else:
        #print("max pair: ", max_pair)
        max_index = ids.index(max_pair[1])
        #print("tod.näk.:n indeksi: ", max_index)
        for item in tie:
            #print("vertailtavan indeksi: ", ids.index(item[1]), " vertailtava: ", item[1])
            #print(f"{ids.index(item[1])} < {max_index}")
            comp_indx = ids.index(item[1])
            if comp_indx < max_index:
                max_pair = item
                max_index = comp_indx
        #print("palautetaan ", max_pair[1])
        #print("\n------------------\n")
        return max_pair[1]

def get_average_face(training_images):
    return np.mean(training_images, axis=1)

def get_coordinates(image: np.array, basis: np.array, average_im: np.array):
    """Calculates coordinates of a given image in the given face space.
    The basis must be orthonormal

    Args:
        image (np.array): image
        space (np.array): eigenface vectors spanning a face space
        average_im (np.array): average image

    Returns:
        np.array: coordinates
    """
    image_diff = image - average_im
    ims = basis.shape[1]
    coordinates_list = []
    for i in range(ims):
        mult = np.dot(image_diff, basis[:,i])
        coordinates_list.append(mult)
    coordinates = np.array(coordinates_list)
    return coordinates

def get_projection(image_crds: np.array, basis: np.array, av_face: np.array) -> np.array:
    """Returns the image contstructed from the given coordinates

    Args:
        image_crds (np.array): image coordinates
        basis (np.array): basis
        av_face (np.array): average image

    Returns:
        np.array: image formed by the linear combination of image_crds
        and basis vectors
    """
    result = np.dot(basis, image_crds)
    return result + av_face