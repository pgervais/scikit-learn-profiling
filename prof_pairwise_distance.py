"""Profiling of distance minimum distance computation"""
import utils  # define profile() when not available
import time

import sys
import numpy as np
import scipy.spatial.distance as ssd

from sklearn.metrics import euclidean_distances
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
            batch_size_x=None, batch_size_y=None, **kwargs):
    """Return argmin on the selected axis.
    axis 0 is along X
    axis 1 is along Y
    """
    out = pairwise_distances_argmin(X, Y=Y, axis=axis,
                                    batch_size_x=batch_size_x,
                                    batch_size_y=batch_size_y,
                                    metric=metric, **kwargs)
    return out


def chunked_timing(X, Y=None, axis=1, metric="euclidean", **kwargs):
    sizes = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    t0 = time.time()
    original(X, Y=None, axis=axis, metric="euclidean", **kwargs)
    t1 = time.time()
    original_timing = t1 - t0

    chunked_timings = []

    for batch_size in sizes:
        print("batch_size: %d" % batch_size)
        t0 = time.time()
        chunked(X, Y=None, axis=axis, metric="euclidean",
                batch_size_x=batch_size, batch_size_y=batch_size, **kwargs)
        t1 = time.time()
        chunked_timings.append(t1 - t0)

    import pylab as pl
    pl.semilogx(sizes, chunked_timings, '-+', label="chunked")
    pl.hlines(original_timing, sizes[0], sizes[-1],
              color='k', label="original")
    pl.grid()
    pl.xlabel("batch size")
    pl.ylabel("execution time (wall clock)")
    pl.title("%d / %d" % (X.shape[0], Y.shape[0]))
    pl.legend()
    pl.show()

if __name__ == "__main__":
    np.random.seed(1)
#    x_size, y_size = 10000, 50
#    x_size, y_size = 100000, 500
#    x_size, y_size = 10000, 5000
#    x_size, y_size = 10000, 10000
#    x_size, y_size = 1000, 50000
#    x_size, y_size = 50, 200000
#    x_size, y_size = 50, 20
#    x_size, y_size = 500, 200
#    x_size, y_size = 500, 2000
    x_size, y_size = 5000, 2000
#    x_size, y_size = 2000, 5000

    n_var = 200
    metric = "euclidean"
    #    metric = 'manhattan'
    #    metric = ssd.canberra
    kwargs = {"squared": False}
    X = np.random.rand(x_size, n_var)
    Y = np.random.rand(y_size, n_var)
    axis = 0
    batch_size_x = 500
    batch_size_y = 500

#    chunked_timing(X, Y, metric=metric, axis=axis, **kwargs)
#    sys.exit(0)

    dist_orig_ind, dist_orig_val = utils.timeit(profile(original))(
        X, Y, metric=metric, axis=axis, **kwargs)
    dist_chunked_ind, dist_chunked_val = utils.timeit(profile(chunked))(
        X, Y, axis=axis, metric=metric,
        batch_size_x=batch_size_x, batch_size_y=batch_size_y, **kwargs)

    np.testing.assert_almost_equal(dist_orig_ind, dist_chunked_ind, decimal=7)
    np.testing.assert_almost_equal(dist_orig_val, dist_chunked_val, decimal=7)
