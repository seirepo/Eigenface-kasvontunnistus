import numpy as np

def sum_matrices(a, b):
    return a + b

def matrix3d_submatrices_to_colums(im_set_matrix):
    """
    Function to turn n*n matrices in an matrix
    to column vectors of length n*n

    Args:
        im_set_matrix (np.array): 3-dim np.array containing images as np.array

    Raises:
        ValueError: if arg is not a 3-dim np.array

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

def matrix2d_to_column_vector(im_matrix):
    """Static function to turn 2-dim np.arrays to a vector.
    Probably not necessary

    Args:
        im_matrix (np.array): 2-dim np.array

    Returns:
        np.array: given arg as a column vector
    """
    return np.array([im_matrix.flatten()]).T