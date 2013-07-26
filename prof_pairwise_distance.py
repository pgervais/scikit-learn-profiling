"""Profiling of distance minimum distance computation"""
import utils  # define profile() when not available

import sys
import numpy as np
import scipy.spatial.distance as ssd

from sklearn.metrics import euclidean_distances
from sklearn.metrics import euclidean_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin


def original(X, Y=None, axis=1, metric='euclidean', **kwargs):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    if metric == "euclidean":
        dist = euclidean_distances(X, Y=Y, **kwargs)
    else:
        dist = pairwise_distances(X, Y=Y, metric=metric, **kwargs)
    indices = dist.argmin(axis=axis)
    if axis == 1:
        return indices, dist[range(len(indices)), indices]
    else:
        return indices, dist[indices, range(len(indices))]


def chunked(X, Y=None, axis=1, metric="euclidean",
            chunk_x_num=None, chunk_y_num=None, **kwargs):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    out = pairwise_distances_argmin(X, Y=Y, axis=axis,
                                    chunk_x_num=chunk_x_num,
                                    chunk_y_num=chunk_y_num,
                                    metric=metric, return_distances=True,
                                    **kwargs)
    return out


def compare_implementations(X, Y=None, axis=1,
                            chunk_x_num=None, chunk_y_num=None):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    utils.timeit(euclidean_distances_argmin)(X, Y=Y, axis=axis,
                                             chunk_x_num=chunk_x_num,
                                             chunk_y_num=chunk_y_num,
                                             squared=False,
                                             return_distances=True)
    utils.timeit(pairwise_distances_argmin)(X, Y=Y, axis=axis,
                                            chunk_x_num=chunk_x_num,
                                            chunk_y_num=chunk_y_num,
                                            metric="euclidean",
                                            return_distances=True)


if __name__ == "__main__":
    np.random.seed(1)
#    x_size, y_size = 200000, 50
#    x_size, y_size = 100000, 500
#    x_size, y_size = 10000, 5000
#    x_size, y_size = 10000, 10000
#    x_size, y_size = 1000, 50000
#    x_size, y_size = 50, 200000
#    x_size, y_size = 50, 20
#    x_size, y_size = 500, 200
#    x_size, y_size = 500, 2000
    x_size, y_size = 5000, 2000
    x_size, y_size = 2000, 5000

    n_var = 200
    metric = "euclidean"
    #    metric = 'manhattan'
    #    metric = ssd.canberra
    kwargs = {"squared": False}
    X = np.random.rand(x_size, n_var)
    Y = np.random.rand(y_size, n_var)
    axis = 0
    chunk_x_num = None
    chunk_y_num = None
    #    batch_size # mini batch kmean

#    compare_implementations(X, Y, axis=axis)
#    sys.exit(0)
    dist_orig_ind, dist_orig_val = utils.timeit(profile(original))(
        X, Y, metric=metric, axis=axis, **kwargs)
    dist_chunked_ind, dist_chunked_val = utils.timeit(profile(chunked))(
        X, Y, axis=axis, metric=metric,
        chunk_x_num=chunk_x_num, chunk_y_num=chunk_y_num, **kwargs)

    np.testing.assert_almost_equal(dist_orig_ind, dist_chunked_ind, decimal=7)
    np.testing.assert_almost_equal(dist_orig_val, dist_chunked_val, decimal=7)
