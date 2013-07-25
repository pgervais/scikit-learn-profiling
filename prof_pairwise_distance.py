"""Profiling of distance minimum distance computation"""
import utils  # define profile() when not available

import numpy as np
from sklearn.metrics import euclidean_distances, euclidean_distances_argmin


def original(X, Y=None, axis=1):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    dist = euclidean_distances(X, Y=Y, squared=True)
    indices = dist.argmin(axis=axis)
    if axis == 1:
        return indices, dist[range(len(indices)), indices]
    else:
        return indices, dist[indices, range(len(indices))]


def chunked(X, Y=None, axis=1, chunk_x_num=None, chunk_y_num=None):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    return euclidean_distances_argmin(X, Y=Y, axis=axis,
                                      chunk_x_num=chunk_x_num,
                                      chunk_y_num=chunk_y_num,
#                                     return_values=True)
                                      return_distances=True, squared=True)


if __name__ == "__main__":
    np.random.seed(1)
#    x_size, y_size = 200000, 50  # chunked slower
#    x_size, y_size = 100000, 500  # draw
#    x_size, y_size = 10000, 5000  # chunked much faster
#    x_size, y_size = 10000, 10000  # chunked much faster
#    x_size, y_size = 1000, 50000  # chunked much faster
    x_size, y_size = 50, 200000  # chunked much faster
#    x_size, y_size = 50, 20  # chunked slower
#    x_size, y_size = 500, 200  # chunked slower
#    x_size, y_size = 5000, 2000  # chunked slower
#    x_size, y_size = 2000, 5000  #

    n_var = 200
    X = np.random.rand(x_size, n_var)
    Y = np.random.rand(y_size, n_var)
    axis = 0

    dist_orig_ind, dist_orig_val = utils.timeit(profile(original))(
        X, Y, axis=axis)
    dist_chunked_ind, dist_chunked_val = utils.timeit(profile(chunked))(
        X, Y, axis=axis, chunk_x_num=None, chunk_y_num=None)

    np.testing.assert_almost_equal(dist_orig_ind, dist_chunked_ind, decimal=7)
    np.testing.assert_almost_equal(dist_orig_val, dist_chunked_val, decimal=7)
