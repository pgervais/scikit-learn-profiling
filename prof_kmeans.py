import utils

import numpy as np
from scipy.cluster.vq import kmeans, kmeans2
from sklearn.cluster import k_means, MiniBatchKMeans
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.datasets import make_blobs


# Having unit standard deviations is very important for a fair
# test between scipy and sklearn: stopping threshold is computed
# differently in both implementations (and are identical if all variances
# are equal to one.
n_clusters = 10

np.random.seed(0)

n_samples, n_features = 100000, 200
X = make_blobs(n_samples, n_features, centers=n_clusters, random_state=0)[0]

## n_samples, n_features = 10000, 200    # scipy faster
## n_samples, n_features = 200, 10000    # scipy faster
## n_samples, n_features = 200, 100000   # scipy faster, minibatch very long
## X = np.random.normal(size=(n_samples, n_features))

tol = 1e-4

print("\n-- scipy.cluster.vq")
ratio = 1.
np.random.seed(1)
sc, _ = utils.timeit(profile(kmeans))(X, n_clusters, iter=2, thresh=tol / ratio)
inertia1 = _labels_inertia(X, (X ** 2).sum(axis=-1), sc)[1]
print('scipy inertia: %.1f' % inertia1)

# Unfair comparison with kmeans2 (no way to specify a tolerance yet)
## print("\n-- scipy.cluster.vq")
## sc, _ = utils.timeit(kmeans2)(X, n_clusters, iter=50)
## print('scipy inertia: %.1f' % _labels_inertia(X, (X ** 2).sum(axis=-1), sc)[1])

print("\n-- sklearn.cluster")
ratio = np.mean(np.var(X, axis=0))  # just to make the comparison fair.
np.random.seed(1)
sk, _, _ = utils.timeit(profile(k_means))(X, n_clusters, n_init=2, tol=tol / ratio,
                                 init="random", random_state=1)
inertia2 = _labels_inertia(X, (X ** 2).sum(axis=-1), sk)[1]
print('inertia: %.1f' % inertia2)

## print ('\nsklearn - scipy inertia: %.1f. Relative variation: %.1e' %
##        ((inertia2 - inertia1), (inertia2 - inertia1) / (
##            2. * (inertia1 + inertia2))))

## print("\n-- sklearn minibatch")
## mbk = utils.timeit(MiniBatchKMeans(n_clusters=n_clusters,
##                                    n_init=2, tol=tol).fit)(X)
## print('inertia: %.1f' % _labels_inertia(X, (X ** 2).sum(axis=-1),
##                                         mbk.cluster_centers_)[1])
