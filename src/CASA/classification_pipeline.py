import numpy as np
from numpy import linalg as LA

import sys

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering

from concurrent.futures import TimeoutError
from pebble import ProcessPool
import multiprocessing

import seaborn as sns

import logging
from progress.bar import Bar


# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def init_worker(X, X_shape):
    """
    Using a dictionary is not strictly necessary.
    We could also use global variables.

    Parameters
    ----------

    X : np.ndarray
        Matrix
    X_shape : tuple
        Shape of the matrix
    """
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def get_entry_distance_matrix(inds):
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
    first = inds[0]
    second = inds[1]
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    first = X_np[first]
    second = X_np[second]
    d = cdist(first, second)
    assignment = linear_sum_assignment(d)
    emd_val = d[assignment].sum()
    return emd_val


def compute_distance_matrix(data_matrix, timeout_time=20, chunksize=1000):
    """
    Compute distance matrix multiprocessing the full computation.
    Wrapper to avoid having to deal with the RawArrays manually.

    Parameters
    ----------

    data_matrix : np.ndarray
        Array of data tensors

    Returns
    ----------
    distance_matrix : np.ndarray
        EMD full distance matrix
    """
    assert(len(data_matrix.shape) == 2 or len(data_matrix.shape) == 3)
    if len(data_matrix.shape) == 2:
        data_matrix = data_matrix[:, np.newaxis]

    X_shape = data_matrix.shape
    X = multiprocessing.RawArray('d', int(np.prod(X_shape)))
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    # Copy data to our shared array.
    np.copyto(X_np, data_matrix)

    # 1. Compute distance matrix by multiprocessing
    distance_matrix = _compute_distance_matrix(data_matrix, X, X_shape, chunksize)
    return distance_matrix


def _compute_distance_matrix(matrix_arrays, X, X_shape, timeout_time=20, chunksize=1000):
    """
    Compute distance matrix multiprocessing the row computation.

    Parameters
    ----------

    matrix_arrays : np.ndarray
        Matrix of data tensors stored in arrays
    X : multiprocessing.RawArray
        RawArray to share matrix_arrays contents between processes
    X_shape : tuple
        Size of the matrix

    Returns
    ----------
    np_arr : np.ndarray
        EMD full distance matrix
    """
    # Get indices for the upper-triangle of matrix array
    indx, indy = np.triu_indices(len(matrix_arrays))
    np_arr = np.zeros((len(matrix_arrays), len(matrix_arrays)))
    bar = Bar('Calculating distance matrix... ', max=len(indx))
    with ProcessPool(max_workers=multiprocessing.cpu_count(),
                     initializer=init_worker,
                     initargs=(X, X_shape)) as pool:
        # iterator
        result = pool.map(get_entry_distance_matrix, zip(indx, indy),
                          chunksize=chunksize, timeout=timeout_time)
        iterator = result.result()
        vals = []
        processed = 0
        while True:
            try:
                val = next(iterator)
                processed = processed + 1
                bar.next()
                vals.append(val)
            except StopIteration:
                break
            except TimeoutError as error:
                logger.info(" Processing EMD took longer than %d s" % error.args[1])
                # This is a hack for now: in the (extremely rare)
                # case that the calculation freezes, we use a generic
                # distance of 1000.
                vals.append(1000.)

    bar.finish()
    ##
    # Construct lower-triangle (it is a symmetric matrix)
    np_arr[indx, indy] = np.array(vals)
    i_lower = np.tril_indices(len(matrix_arrays), -1)
    np_arr[i_lower] = np_arr.T[i_lower]
    logger.info(" Constructed entire distance matrix")
    return np_arr

def _compute_distance_matrix_subset(matrix_arrays, X, X_shape, indx, indy, timeout_time=20, chunksize = 1000):
    """
    Compute distance matrix multiprocessing the row computation.
    Only the subset of indices given by indx/indy are calculated.

    Parameters
    ----------

    matrix_arrays : np.ndarray
        Matrix of data tensors stored in arrays
    X : multiprocessing.RawArray
        RawArray to allow matrix_arrays contents to be shared between processes.
    X_shape : tuple
        Size of the matrix
    indx : np.ndarray
        Indices along axis 0 to calculate
    indy : np.ndarray
        Indices along axis 1 to calculate
    Returns
    ----------
    np_arr : np.ndarray
        EMD distance matrix
    """
    np_arr = np.zeros((len(matrix_arrays), len(matrix_arrays)))
    bar = Bar('Calculating distance matrix... ', max=len(indx))

    with ProcessPool(max_workers=multiprocessing.cpu_count(),
                     initializer=init_worker,
                     initargs=(X, X_shape)) as pool:
        result = pool.map(get_entry_distance_matrix, zip(indx, indy),
                          chunksize=chunksize, timeout=timeout_time)
        iterator = result.result()
        vals = []
        processed = 0
        while True:
            try:
                val = next(iterator)
                processed = processed + 1
                bar.next()
                vals.append(val)
            except StopIteration:
                break
            except TimeoutError as error:
                logger.info(" Processing distance took longer than %d seconds" % error.args[1])
                # This is a hack for now: in the (extremely rare)
                # case that the calculation freezes, we use a generic
                # distance of 1000.
                vals.append(1000.)
    ##
    bar.finish()
    np_arr[indx, indy] = np.array(vals)
    logger.info(" Construted subset of distance matrix")
    return np_arr


def classify_new_examples(classified_array, classes, unclassified_array):
    """
    Determine the clusters of new examples by using a previously clustered reference data set.

    Parameters
    ----------

    classified_array : np.ndarray
        Data array of already clustered examples.
    classes : np.ndarray
        Clusters/classes of the already clustered examples.
    unclassified_array : np.ndarray
        Data array of unclustered examples.

    Returns
    ----------
    new_classes : np.ndarray
        Clusters/classes of unclassified_array.
    """

    assert(classified_array.shape[0] == classes.shape[0])

    full_array = np.concatenate((classified_array,unclassified_array), axis=0)

    assert( len(full_array.shape) == 2 or len(full_array.shape) == 3)
    if len(full_array.shape) == 2:
        full_array = full_array[:,np.newaxis]

    X_shape = full_array.shape
    X = multiprocessing.RawArray('d', int(np.prod(X_shape)))
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    # Copy data to our shared array.
    np.copyto(X_np, full_array)
    inds_old = np.array(list(range(len(classified_array))))
    inds_new = np.array(list(range(len(unclassified_array)))) + len(classified_array)
    indx, indy = np.meshgrid(inds_old, inds_new)
    indx = indx.flatten()
    indy = indy.flatten()
    logger.info("Calculating distance matrix...")
    dist_matrix = _compute_distance_matrix_subset(full_array, X, X_shape, indx, indy)
    logger.info("Distance matrix calculated.")
    # Find the closest example in the already classified data set
    # TODO: Implement alternative schemes for determining classification?
    new_classes_ind = np.argmin( dist_matrix[:len(classified_array),len(classified_array):], axis=0 ) 
    new_classes = classes[new_classes_ind]
    return new_classes


def get_affinity_matrix(distance_matrix, k=7):
    """
    Compute affinity matrix based on distance matrix and 
    apply local scaling based on the k nearest neighbour.

    Parameters
    ----------

    distance_matrix : np.ndarray
        Distance matrix (symmetric matrix)
    k : int
        k-th position for local statistics of the neighborhood
        surrounding each point (local scaling parameter for each
        point allows self-tuning of the point-to-point distances)

    Returns
    ----------
    affinity_matrix : np.ndarray
        affinity_matrix

    References
    ----------
    .. [1] https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    .. [2] code adapted from: 
           https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
    """
    # distance matrix (n x n)
    dists = distance_matrix
    # for each row, sort the distances ascendingly and take the index of the
    # k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    # create column vector (n x 1)
    knn_distances = knn_distances[np.newaxis].T
    # create a simmeric matrix (n x n) with rank=1
    # by column times row vectors moltiplication
    local_scale = knn_distances.dot(knn_distances.T)
    # divide square distance matrix by local scale
    affinity_matrix = dists * dists
    affinity_matrix = - affinity_matrix / local_scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigen_decomposition(A, topK=5):
    """
    Compute optimum number of clusters for Spectral Clustering based on eigengap heuristic algorithm.

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters by eigengap heuristic

    Parameters
    ----------

    A : np.ndarray
        affinity matrix
    topK : int
        top optimal number of clusters to return

    Returns
    ----------
    nb_clusters : list[int]
        topK optimal numbers of clusters by eigengap heuristic


    References
    ----------
    .. [1] https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    .. [2] http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh),
    # that is, largest eigenvalues in  the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = LA.eig(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigenvalues
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters


def get_clusters_spectral(distance_matrix, ncl=0, dlt=0, self_tuned=False):
    """
    Compute clusters with Spectral Clustering on a precomputed distance matrix.

    Parameters
    ----------

    distance_matrix : np.ndarray
        distance matrix
    ncl : int
        number of clusters
    dlt : float
        delta parameter
    k : int
        k-th position for local statistics of the neighborhood
    self_tuned : bool
        activate self-tuning

    Returns
    ----------
     tuple : tuple[np.ndarray, list, sklearn.cluster._spectral.SpectralClustering]
        - cl_labels is an array of clusters labels
        - cl_colors is an list of clusters colors
        - clusterer is the clusterer
    """
    if self_tuned:
        affinity_matrix = get_affinity_matrix(distance_matrix)
        if ncl == 0:
            k = eigen_decomposition(affinity_matrix)
            ncl = k[0]
            logger.info(f" Optimal n. of clusters {ncl}")
            logger.info("*" * 40)
        clusterer = SpectralClustering(n_clusters=ncl, affinity="precomputed").fit(affinity_matrix)

    else:
        if ncl == 0:
            logger.error("*" * 40)
            logger.error(" ncl cannot be zero unless self_tuned is True.")
            logger.error("*" * 40)
            sys.exit(1)
        if dlt == 0:
            logger.error("*" * 40)
            logger.error(" delta cannot be zero unless self_tuned is True.")
            logger.error("*" * 40)
            sys.exit(1)
        affinity_matrix = np.exp(- distance_matrix ** 2 / (2. * dlt ** 2))
        clusterer = SpectralClustering(n_clusters=ncl, affinity="precomputed").fit(affinity_matrix)

    cl_labels = clusterer.labels_
    palette = sns.color_palette('deep', np.unique(cl_labels).max() + 1)
    cl_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in cl_labels]
    return cl_labels, cl_colors, clusterer


def get_clusters_hdbscan(par_1, par_2, par_3, par_4, distance_matrix):
    """
    Compute clusters with HDBSCAN on a precomputed distance matrix between
    wavelet spectra obtained from RO profiles.

    Parameters
    ----------
    par_1 : int
        min_cluster_size - The minimum size of clusters; single linkage splits that contain fewer points
        than this will be considered points "falling out" of a cluster rather than
        a cluster splitting into two new clusters.
    par_2 : int
        min_samples - The number of samples in a neighbourhood for a point to be considered a core point.
    par_3 : string
        The method used to select clusters from the condensed tree.
        The standard approach for HDBSCAN is to use an Excess of Mass algorithm to find the most persistent clusters.
        Alternatively you can instead select the clusters at the leaves of the tree – this provides
        the most fine grained and homogeneous clusters. Options are: eom and leaf. Default is eom
    par_4 : float
        cluster_selection_method - A distance threshold. Clusters below this value will be merged.
    distance_matrix : np.ndarray
        Precomputed distance matrix

    Returns
    ----------
     tuple : tuple[np.ndarray, list, sklearn.cluster._spectral.SpectralClustering]
        - cl_labels is an array of clusters labels
        - cl_colors is an list of clusters colors
        - clusterer is the clusterer
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                min_cluster_size=par_1,
                                min_samples=par_2,
                                cluster_selection_method=par_3,
                                cluster_selection_epsilon=par_4)

    clusterer.fit(distance_matrix)

    cl_labels = clusterer.labels_
    palette = sns.color_palette('deep', np.unique(cl_labels).max() + 1)
    cl_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in cl_labels]
    logging.info(f"N. of clusters with HDBSCAN: { len(np.unique(cl_labels)) }")
    return cl_labels, cl_colors, clusterer

