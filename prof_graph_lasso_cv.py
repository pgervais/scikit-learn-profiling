import numpy as np
from scipy import linalg

from sklearn.utils import check_random_state
from sklearn.datasets.samples_generator import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV
from sklearn.cross_validation import KFold

import utils


def prof_graph_lasso_cv(random_state_seed=1):
    # Sample data from a sparse multivariate normal
    dim = 10  # 80
    n_samples = 60

    # Generate input data
    random_state = check_random_state(random_state_seed)
    prec = make_sparse_spd_matrix(dim, alpha=.96, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)

    utils.cache_value(X, "prof_graph_lasso_cv/X_%d_%d_%d" %
                      (dim, n_samples, random_state_seed))

    # Test with alphas as integer
    ## mode = 'cd'
    ## gl1 = utils.timeit(GraphLassoCV(verbose=1, alphas=3, mode=mode).fit)(X)
    ## utils.cache_value(gl1.covariance_,
    ##                   "prof_graph_lasso_cv/covariance_%d_%d_%d" %
    ##                   (dim, n_samples, random_state_seed))
    ## utils.cache_value(gl1.precision_,
    ##                   "prof_graph_lasso_cv/precision_%d_%d_%d" %
    ##                   (dim, n_samples, random_state_seed))

    # Test with alphas as list.
    # Take same alphas as were found in the first step, check the result
    # is the same.
    ## gl2 = utils.timeit(GraphLassoCV(alphas=gl1.cv_alphas_, n_jobs=1,
    ##                                 mode=mode).fit)(X)
    ## np.testing.assert_almost_equal(gl1.covariance_, gl2.covariance_,
    ##                                decimal=3)
    ## np.testing.assert_almost_equal(gl1.precision_, gl2.precision_,
    ##                                decimal=3)
    ## np.testing.assert_almost_equal(gl1.alpha_, gl2.alpha_)

    # Smoke test with an alternate cross-validation object.
    gl3 = utils.timeit(GraphLassoCV(cv=KFold(n=X.shape[0], n_folds=20),
                                    n_jobs=1).fit)(X)
    ## utils.cache_value(gl3.covariance_,
    ##                   "prof_graph_lasso_cv/gl3_covariance_%d_%d_%d" %
    ##                   (dim, n_samples, random_state_seed))
    ## utils.cache_value(gl3.precision_,
    ##                   "prof_graph_lasso_cv/gl3_precision_%d_%d_%d" %
    ##                   (dim, n_samples, random_state_seed))

if __name__ == "__main__":
    prof_graph_lasso_cv(2)
#itertools.product
# (3)
## inverse, no restart: 60.8 / 59.7/
## direct, no restart: 59.9 / 59.8
## inverse, restart: 52.8 / 53.5
## direct, restart: 38.8 / 37.8
## direct, double restart: 37.9 / 38 ; 38.1 / 37.8

# (1)
## direct, restart: 55.0 / 72.6
## direct, double restart: 60.0 / 72.5

# (2)
## direct, restart: 39.2 / 43.8
## direct, double restart: 37.1 / 44.3
## direct, restart, simplified: 39.1 / 43.9

## (2), lars: doesn't work

## direct, restart:  /
## direct, double restart:  /
