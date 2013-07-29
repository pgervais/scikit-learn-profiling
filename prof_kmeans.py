import utils

import numpy as np
from scipy.cluster.vq import kmeans
from sklearn.cluster import k_means
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.datasets import make_blobs


# Having unit standard deviations is very important for a fair
# test between scipy and sklearn: stopping threshold is computed
# differently in both implementations (and are identical if all variances
# are equal to one.
n_clusters = 10

random_state = 4

n_samples, n_features = 100000, 200
#n_samples, n_features = 200, 10000
X = make_blobs(n_samples, n_features, centers=n_clusters, random_state=0)[0]

## n_samples, n_features = 10000, 200
## n_samples, n_features = 200, 10000
## n_samples, n_features = 200, 100000
## X = np.random.normal(size=(n_samples, n_features))

tol = 1e-4

## print("\n-- scipy.cluster.vq")
## ratio = 1.
## np.random.seed(random_state)
## sc, _ = utils.timeit(profile(kmeans))(X, n_clusters, iter=2,
##                                       thresh=tol / ratio)
## ## utils.cache_value(sc, 'prof_kmeans/scipy_kmeans_%d_%d'
## ##                   % (n_samples, n_features))
## inertia1 = _labels_inertia(X, (X ** 2).sum(axis=-1), sc)[1]
## print('scipy inertia: %.1f' % np.sqrt(inertia1))

print("\n-- sklearn.cluster")
ratio = 1. #np.mean(np.var(X, axis=0))  # just to make the comparison fair.

np.random.seed(random_state)
sk, _, _ = utils.timeit(profile(k_means))(X, n_clusters, n_init=2,
                                          tol=tol / ratio,
                                          init="random",
                                          random_state=random_state)
## utils.cache_value(sk, 'prof_kmeans/sklearn_kmeans_%d_%d' %
##                   (n_samples, n_features))
inertia2 = _labels_inertia(X, (X ** 2).sum(axis=-1), sk)[1]
print('inertia: %.1f' % np.sqrt(inertia2))

## print ('\nsklearn - scipy inertia: %.1f. Relative variation: %.1e' %
##        ((inertia2 - inertia1), (inertia2 - inertia1) / (
##            2. * (inertia1 + inertia2))))
