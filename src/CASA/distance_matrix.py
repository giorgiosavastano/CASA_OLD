import logging

import dask.bag as db
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger("casa-classification")

# from dask.distributed import Client
# client = Client(n_workers=4)


def compute_earth_mover_dist(first, second):
    """
    Compute single entry of the distance matrix.

    Parameters
    ----------
    inds : tuple
        Indexes of the first and second arrays

    Returns
    ----------
    emd_val : float
        EMD distance between the two arrays
    """
    d = cdist(first, second)
    assignment = linear_sum_assignment(d)
    emd_val = d[assignment].sum()
    return emd_val


def compute_distance_matrix(matrix_arrays, num_part=-1):
    """Compute distance matrix using Dask.

    Author: Giorgio Savastano (giorgiosavastano@gmail.com)

    Parameters
    ----------
    matrix_arrays : np.ndarray
        Matrix of data tensors stored in arrays.
        Only 1-D or 2-D data tensors allowed
    num_part : int, optional
        Number of partitions for Dask bags.

    """
    # Get indices for the upper-triangle of matrix array
    indx, indy = np.triu_indices(len(matrix_arrays))
    np_arr = np.zeros((len(matrix_arrays), len(matrix_arrays)))

    if len(matrix_arrays.shape) == 2:
        # for a matrix of 1-D data tensors
        arr_1 = matrix_arrays[indx][:, np.newaxis]
        arr_2 = matrix_arrays[indy][:, np.newaxis]

    elif len(matrix_arrays.shape) == 3:
        # for a matrix of 2-D data tensors
        arr_1 = matrix_arrays[indx]
        arr_2 = matrix_arrays[indy]

    else:
        logger.error(" Distance matrix can be compute on 1-D and 2-D data tensors only")
        raise ValueError

    if num_part == -1:
        num_part = int(arr_1.shape[0] / 100)

    logger.info(f"Number of partitions = {num_part}")

    b1 = db.from_sequence(arr_1, npartitions=num_part)
    b2 = db.from_sequence(arr_2, npartitions=num_part)

    results = db.map(compute_earth_mover_dist, first=b1, second=b2).compute()

    np_arr[indx, indy] = np.array(results)
    # Construct lower-triangle (it is a symmetric matrix)
    i_lower = np.tril_indices(len(matrix_arrays), -1)
    np_arr[i_lower] = np_arr.T[i_lower]
    logger.info(" Constructed entire distance matrix")

    return np_arr
