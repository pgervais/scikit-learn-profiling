"""Profiling of distance minimum distance computation"""
import utils  # define profile() when not available
import sys
import time

import numpy as np
import scipy.spatial.distance as ssd

from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min


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


def chunked(X, Y, axis=1, metric="euclidean", batch_size=500, **kwargs):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    return pairwise_distances_argmin_min(X, Y, axis=axis,
                                         batch_size=batch_size,
                                         metric=metric, **kwargs)


def chunked_timing(X, Y, axis=1, metric="euclidean", **kwargs):
    sizes = [20, 50, 100,
             200, 500, 1000,
             2000, 5000, 10000,
             20000, 50000, 100000,
             200000]

    t0 = time.time()
    original(X, Y, axis=axis, metric=metric, **kwargs)
    t1 = time.time()
    original_timing = t1 - t0

    chunked_timings = []

    for batch_size in sizes:
        print("batch_size: %d" % batch_size)
        t0 = time.time()
        chunked(X, Y, axis=axis, metric=metric, batch_size=batch_size,
                **kwargs)
        t1 = time.time()
        chunked_timings.append(t1 - t0)

    import pylab as pl
    pl.semilogx(sizes, chunked_timings, '-+', label="chunked")
    pl.hlines(original_timing, sizes[0], sizes[-1],
              color='k', label="original")
    pl.grid()
    pl.xlabel("batch size")
    pl.ylabel("execution time (wall clock)")
    pl.title("%s %d / %d (axis %d)" % (
        str(metric), X.shape[0], Y.shape[0], axis))
    pl.legend()
    pl.savefig("%s_%d_%d_%d" % (str(metric), X.shape[0], Y.shape[0], axis))
    pl.show()

if __name__ == "__main__":
    np.random.seed(1)
#    x_size, y_size = 10000, 50
#    x_size, y_size = 100000, 500
#    x_size, y_size = 10000, 5000
#    x_size, y_size = 10000, 10000
#    x_size, y_size = 1000, 50000
#    x_size, y_size = 50, 200000
    x_size, y_size = 200000, 50
#    x_size, y_size = 50, 20
#    x_size, y_size = 500, 200
#    x_size, y_size = 500, 2000
#    x_size, y_size = 5000, 2000
#    x_size, y_size = 2000, 5000
#    x_size, y_size = 1500, 50
#    x_size, y_size = 50, 1500

    n_var = 200
    metric = "euclidean"
#    metric = "chebyshev"
    # metric = "mahalanobis" # doesn't work
    # metric = 'manhattan'
    # metric = "minkowski"

    if metric == "euclidean":
        kwargs = {"squared": False}
    elif metric == "minkowski":
        kwargs = {"p": 2}
    else:
        kwargs = {}
    X = np.random.rand(x_size, n_var)
    Y = np.random.rand(y_size, n_var)
    axis = 1
    batch_size = 200

#    chunked_timing(X, Y, metric=metric, axis=axis, **kwargs); sys.exit(0)

    dist_orig_ind, dist_orig_val = utils.timeit(profile(original))(
        X, Y, metric=metric, axis=axis, **kwargs)
    dist_chunked_ind, dist_chunked_val = utils.timeit(profile(chunked))(
        X, Y, axis=axis, metric=metric,
        batch_size=batch_size, **kwargs)

    np.testing.assert_almost_equal(dist_orig_ind, dist_chunked_ind, decimal=7)
    np.testing.assert_almost_equal(dist_orig_val, dist_chunked_val, decimal=7)
