# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:39:55 2021

@author: liuy47
"""

#%%Config
#jumap_
from __future__ import print_function

#__init__
#from .jumap_ import UMAP, JUMAPBASE, JUMAP
#from ._t_sne import TSNE, JTSNEBASE, JTSNE
import numba
import pkg_resources


import locale
import warnings
from warnings import warn
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph

try:
    # Use pynndescent, if installed (python 3 only)
    from pynndescent import NNDescent
    from pynndescent.distances import named_distances as pynn_named_distances
    from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances

    _HAVE_PYNNDESCENT = True
except ImportError:
    _HAVE_PYNNDESCENT = False


#distances
import scipy.stats


#import Jvis.distances as dist
#import Jvis.sparse as sparse
#import Jvis.sparse_nndescent as sparse_nn

#from Jvis.utils import (
#    tau_rand_int,
#    deheap_sort,
#    submatrix,
#    ts,
#    csr_unique,
#    fast_knn_indices,
#    tau_rand,
#    norm,
#    make_heap,
#    heap_push,
#    rejection_sample,
#    build_candidates,
#    unchecked_heap_push,
#    smallest_flagged,
#    new_build_candidates,
#)

#from Jvis.rp_tree import search_sparse_flat_tree, rptree_leaf_array, make_forest
#from Jvis.nndescent import (
#    # make_nn_descent,
#    # make_initialisations,
#    # make_initialized_nnd_search,
#    nn_descent,
#    initialized_nnd_search,
#    initialise_search,
#)

#from Jvis.spectral import spectral_layout
#from Jvis.layouts import (
#    optimize_layout_euclidean,
#    optimize_layout_generic,
#    optimize_layout_inverse,
#)


#rc_tree
from collections import deque, namedtuple

#spectral

from sklearn.manifold import SpectralEmbedding


#_t_sne

from time import time

from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import check_non_negative
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.decomposition import PCA

# mypy error: Module 'hoan.manifold' has no attribute '_utils'
# from  import _utils  # type: ignore
from sklearn.manifold import _utils
# mypy error: Module 'hoan.manifold' has no attribute '_barnes_hut_tsne'
from sklearn.manifold import _barnes_hut_tsne  # type: ignore


#%%main

#distances###################


_mock_identity = np.eye(2, dtype=np.float64)
_mock_cost = 1.0 - _mock_identity
_mock_ones = np.ones(2, dtype=np.float64)


@numba.njit()
def sign(a):
    if a < 0:
        return -1
    else:
        return 1


@numba.njit(fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance.
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    """Standard euclidean distance and its gradient.
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad


@numba.njit()
def standardised_euclidean(x, y, sigma=_mock_ones):
    """Euclidean distance standardised against a vector of standard
    deviations per coordinate.
    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / sigma[i]

    return np.sqrt(result)


@numba.njit(fastmath=True)
def standardised_euclidean_grad(x, y, sigma=_mock_ones):
    """Euclidean distance standardised against a vector of standard
    deviations per coordinate with gradient.
    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2 / sigma[i]
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d * sigma)
    return d, grad


@numba.njit()
def manhattan(x, y):
    """Manhattan, taxicab, or l1 distance.
    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit()
def manhattan_grad(x, y):
    """Manhattan, taxicab, or l1 distance with gradient.
    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])
        grad[i] = np.sign(x[i] - y[i])
    return result, grad


@numba.njit()
def chebyshev(x, y):
    """Chebyshev or l-infinity distance.
    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result = max(result, np.abs(x[i] - y[i]))

    return result


@numba.njit()
def chebyshev_grad(x, y):
    """Chebyshev or l-infinity distance with gradient.
    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    max_i = 0
    for i in range(x.shape[0]):
        v = np.abs(x[i] - y[i])
        if v > result:
            result = v
            max_i = i
    grad = np.zeros(x.shape)
    grad[max_i] = np.sign(x[max_i] - y[max_i])

    return result, grad


@numba.njit()
def minkowski(x, y, p=2):
    """Minkowski distance.
    ..math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def minkowski_grad(x, y, p=2):
    """Minkowski distance with gradient.
    ..math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (np.abs(x[i] - y[i])) ** p

    grad = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        grad[i] = (
            pow(np.abs(x[i] - y[i]), (p - 1.0))
            * sign(x[i] - y[i])
            * pow(result, (1.0 / (p - 1)))
        )

    return result ** (1.0 / p), grad


@numba.njit()
def poincare(u, v):
    """Poincare distance.
    ..math::
        \delta (u, v) = 2 \frac{ \lVert  u - v \rVert ^2 }{ ( 1 - \lVert  u \rVert ^2 ) ( 1 - \lVert  v \rVert ^2 ) }
        D(x, y) = \operatorname{arcosh} (1+\delta (u,v))
    """
    sq_u_norm = np.sum(u * u)
    sq_v_norm = np.sum(v * v)
    sq_dist = np.sum(np.power(u - v, 2))
    return np.arccosh(1 + 2 * (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))))


@numba.njit()
def hyperboloid_grad(x, y):
    s = np.sqrt(1 + np.sum(x ** 2))
    t = np.sqrt(1 + np.sum(y ** 2))

    B = s * t
    for i in range(x.shape[0]):
        B -= x[i] * y[i]

    if B <= 1:
        B = 1.0 + 1e-8

    grad_coeff = 1.0 / (np.sqrt(B - 1) * np.sqrt(B + 1))

    # return np.arccosh(B), np.zeros(x.shape[0])

    grad = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        grad[i] = grad_coeff * (((x[i] * t) / s) - y[i])

    return np.arccosh(B), grad


@numba.njit()
def weighted_minkowski(x, y, w=_mock_ones, p=2):
    """A weighted version of Minkowski distance.
    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (w[i] * np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def weighted_minkowski_grad(x, y, w=_mock_ones, p=2):
    """A weighted version of Minkowski distance with gradient.
    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (w[i] * np.abs(x[i] - y[i])) ** p

    grad = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        grad[i] = (
            w[i] ** p
            * pow(np.abs(x[i] - y[i]), (p - 1.0))
            * sign(x[i] - y[i])
            * pow(result, (1.0 / (p - 1)))
        )

    return result ** (1.0 / p), grad


@numba.njit()
def mahalanobis(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)


@numba.njit()
def mahalanobis_grad(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    grad_tmp = np.zeros(x.shape)
    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
            grad_tmp[i] += vinv[i, j] * diff[j]
        result += tmp * diff[i]
    dist = np.sqrt(result)
    grad = grad_tmp / (1e-6 + dist)
    return dist, grad


@numba.njit()
def hamming(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@numba.njit()
def canberra(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator

    return result


@numba.njit()
def canberra_grad(x, y):
    result = 0.0
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator
            grad[i] = (
                np.sign(x[i] - y[i]) / denominator
                - np.abs(x[i] - y[i]) * np.sign(x[i]) / denominator ** 2
            )

    return result, grad


@numba.njit()
def bray_curtis(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@numba.njit()
def bray_curtis_grad(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        dist = float(numerator) / denominator
        grad = (np.sign(x - y) - dist) / denominator
    else:
        dist = 0.0
        grad = np.zeros(x.shape)

    return dist, grad


@numba.njit()
def jaccard(x, y):
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def matching(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return float(num_not_equal) / x.shape[0]


@numba.njit()
def dice(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def kulsinski(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + x.shape[0]) / (
            num_not_equal + x.shape[0]
        )


@numba.njit()
def rogers_tanimoto(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def russellrao(x, y):
    num_true_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true

    if num_true_true == np.sum(x != 0) and num_true_true == np.sum(y != 0):
        return 0.0
    else:
        return float(x.shape[0] - num_true_true) / (x.shape[0])


@numba.njit()
def sokal_michener(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def sokal_sneath(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def haversine(x, y):
    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    result = np.sqrt(sin_lat ** 2 + np.cos(x[0]) * np.cos(y[0]) * sin_long ** 2)
    return 2.0 * np.arcsin(result)


@numba.njit()
def haversine_grad(x, y):
    # spectral initialization puts many points near the poles
    # currently, adding pi/2 to the latitude avoids problems
    # TODO: reimplement with quaternions to avoid singularity

    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    cos_lat = np.cos(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    cos_long = np.cos(0.5 * (x[1] - y[1]))

    a_0 = np.cos(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long ** 2
    a_1 = a_0 + sin_lat ** 2

    d = 2.0 * np.arcsin(np.sqrt(min(max(abs(a_1), 0), 1)))
    denom = np.sqrt(abs(a_1 - 1)) * np.sqrt(abs(a_1))
    grad = np.array(
        [
            (
                sin_lat * cos_lat
                - np.sin(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long ** 2
            ),
            (np.cos(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long * cos_long),
        ]
    ) / (denom + 1e-6)
    return d, grad


@numba.njit()
def yule(x, y):
    num_true_true = 0.0
    num_true_false = 0.0
    num_false_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_true_false += x_true and (not y_true)
        num_false_true += (not x_true) and y_true

    num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

    if num_true_false == 0.0 or num_false_true == 0.0:
        return 0.0
    else:
        return (2.0 * num_true_false * num_false_true) / (
            num_true_true * num_false_false + num_true_false * num_false_true
        )


@numba.njit()
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))


@numba.njit(fastmath=True)
def cosine_grad(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        dist = 0.0
        grad = np.zeros(x.shape)
    elif norm_x == 0.0 or norm_y == 0.0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        grad = -(x * result - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)
        dist = 1.0 - (result / np.sqrt(norm_x * norm_y))

    return dist, grad


@numba.njit()
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))


@numba.njit()
def hellinger(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    for i in range(x.shape[0]):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return np.sqrt(1 - result / np.sqrt(l1_norm_x * l1_norm_y))


@numba.njit()
def hellinger_grad(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    grad_term = np.empty(x.shape[0])

    for i in range(x.shape[0]):
        grad_term[i] = np.sqrt(x[i] * y[i])
        result += grad_term[i]
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        dist = 0.0
        grad = np.zeros(x.shape)
    elif l1_norm_x == 0 or l1_norm_y == 0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        dist_denom = np.sqrt(l1_norm_x * l1_norm_y)
        dist = np.sqrt(1 - result / dist_denom)
        grad_denom = 2 * dist
        grad_numer_const = (l1_norm_y * result) / (2 * dist_denom ** 3)

        grad = (grad_numer_const - (y / grad_term * dist_denom)) / grad_denom

    return dist, grad


@numba.njit()
def approx_log_Gamma(x):
    if x == 1:
        return 0
    # x2= 1/(x*x);
    return x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi / x) + 1.0 / (x * 12.0)
    # + x2*(-1.0/360.0) + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)  +\
    #  x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 +\
    #  x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0) +\
    #  x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))


@numba.njit()
def log_beta(x, y):
    a = min(x, y)
    b = max(x, y)
    if b < 5:
        value = -np.log(b)
        for i in range(1, int(a)):
            value += np.log(i) - np.log(b + i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


@numba.njit()
def log_single_beta(x):
    return np.log(2.0) * (-2.0 * x + 0.5) + 0.5 * np.log(2.0 * np.pi / x) + 0.125 / x


# + x2*(-1.0/192.0 + x2* (1.0/640.0 + x2*(-17.0/(14336.0) +\
#  x2*(31.0/18432.0 + x2*(-691.0/180224.0 +\
#  x2*(5461.0/425984.0 + x2*(-929569.0/15728640.0 +\
#  x2*(3189151.0/8912896.0 + x2*(-221930581.0/79691776.0) +\
#  x2*(4722116521.0/176160768.0 + x2*(-968383680827.0/3087007744.0 +\
#  x2*(14717667114151.0/3355443200.0 ))))))))))))


@numba.njit()
def ll_dirichlet(data1, data2):
    """ The symmetric relative log likelihood of rolling data2 vs data1
    in n trials on a die that rolled data1 in sum(data1) trials.
    
    ..math::
        D(data1, data2) = DirichletMultinomail(data2 | data1)  
    """

    n1 = np.sum(data1)
    n2 = np.sum(data2)

    log_b = 0.0
    self_denom1 = 0.0
    self_denom2 = 0.0

    for i in range(data1.shape[0]):
        if data1[i] * data2[i] > 0.9:
            log_b += log_beta(data1[i], data2[i])
            self_denom1 += log_single_beta(data1[i])
            self_denom2 += log_single_beta(data2[i])

        else:
            if data1[i] > 0.9:
                self_denom1 += log_single_beta(data1[i])

            if data2[i] > 0.9:
                self_denom2 += log_single_beta(data2[i])

    return np.sqrt(
        1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
    )


@numba.njit(fastmath=True)
def symmetric_kl(x, y, z=1e-11):
    """
    symmetrized KL divergence between two probability distributions
    ..math::
        D(x, y) = \frac{D_{KL}\left(x \Vert y\right) + D_{KL}\left(y \Vert x\right)}{2}
    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x[i] += z
        x_sum += x[i]
        y[i] += z
        y_sum += y[i]

    for i in range(n):
        x[i] /= x_sum
        y[i] /= y_sum

    for i in range(n):
        kl1 += x[i] * np.log(x[i] / y[i])
        kl2 += y[i] * np.log(y[i] / x[i])

    return (kl1 + kl2) / 2


@numba.njit(fastmath=True)
def symmetric_kl_grad(x, y, z=1e-11):
    """
    symmetrized KL divergence and its gradient
    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x[i] += z
        x_sum += x[i]
        y[i] += z
        y_sum += y[i]

    for i in range(n):
        x[i] /= x_sum
        y[i] /= y_sum

    for i in range(n):
        kl1 += x[i] * np.log(x[i] / y[i])
        kl2 += y[i] * np.log(y[i] / x[i])

    dist = (kl1 + kl2) / 2
    grad = (np.log(y / x) - (x / y) + 1) / 2

    return dist, grad


@numba.njit()
def correlation_grad(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        dist = 0.0
        grad = np.zeros(x.shape)
    elif dot_product == 0.0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        dist = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
        grad = ((x - mu_x) / norm_x - (y - mu_y) / dot_product) * dist

    return dist, grad


@numba.njit(fastmath=True)
def sinkhorn_distance(x, y, M=_mock_identity, cost=_mock_cost, maxiter=64):
    p = (x / x.sum()).astype(np.float32)
    q = (y / y.sum()).astype(np.float32)

    u = np.ones(p.shape, dtype=np.float32)
    v = np.ones(q.shape, dtype=np.float32)

    for n in range(maxiter):
        t = M @ v
        u[t > 0] = p[t > 0] / t[t > 0]
        t = M.T @ u
        v[t > 0] = q[t > 0] / t[t > 0]

    pi = np.diag(v) @ M @ np.diag(u)
    result = 0.0
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            if pi[i, j] > 0:
                result += pi[i, j] * cost[i, j]

    return result


@numba.njit(fastmath=True)
def spherical_gaussian_energy_grad(x, y):
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = np.abs(x[2]) + np.abs(y[2])
    sign_sigma = np.sign(x[2])

    dist = (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma) + np.log(sigma) + np.log(2 * np.pi)
    grad = np.empty(3, np.float32)

    grad[0] = mu_1 / sigma
    grad[1] = mu_2 / sigma
    grad[2] = sign_sigma * (1.0 / sigma - (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma ** 2))

    return dist, grad


@numba.njit(fastmath=True)
def diagonal_gaussian_energy_grad(x, y):
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma_11 = np.abs(x[2]) + np.abs(y[2])
    sigma_12 = 0.0
    sigma_22 = np.abs(x[3]) + np.abs(y[3])

    det = sigma_11 * sigma_22
    sign_s1 = np.sign(x[2])
    sign_s2 = np.sign(x[3])

    if det == 0.0:
        # TODO: figure out the right thing to do here
        return mu_1 ** 2 + mu_2 ** 2, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    cross_term = 2 * sigma_12
    m_dist = (
        np.abs(sigma_22) * (mu_1 ** 2)
        - cross_term * mu_1 * mu_2
        + np.abs(sigma_11) * (mu_2 ** 2)
    )

    dist = (m_dist / det + np.log(np.abs(det))) / 2.0 + np.log(2 * np.pi)
    grad = np.empty(6, dtype=np.float32)

    grad[0] = (2 * sigma_22 * mu_1 - cross_term * mu_2) / (2 * det)
    grad[1] = (2 * sigma_11 * mu_2 - cross_term * mu_1) / (2 * det)
    grad[2] = sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2 ** 2) / (2 * det ** 2)
    grad[3] = sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1 ** 2) / (2 * det ** 2)

    return dist, grad


@numba.njit(fastmath=True)
def gaussian_energy_grad(x, y):
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    # Ensure width are positive
    x[2] = np.abs(x[2])
    y[2] = np.abs(y[2])

    # Ensure heights are positive
    x[3] = np.abs(x[3])
    y[3] = np.abs(y[3])

    # Ensure angle is in range -pi,pi
    x[4] = np.arcsin(np.sin(x[4]))
    y[4] = np.arcsin(np.sin(y[4]))

    # Covariance entries for y
    a = y[2] * np.cos(y[4]) ** 2 + y[3] * np.sin(y[4]) ** 2
    b = (y[2] - y[3]) * np.sin(y[4]) * np.cos(y[4])
    c = y[3] * np.cos(y[4]) ** 2 + y[2] * np.sin(y[4]) ** 2

    # Sum of covariance matrices
    sigma_11 = x[2] * np.cos(x[4]) ** 2 + x[3] * np.sin(x[4]) ** 2 + a
    sigma_12 = (x[2] - x[3]) * np.sin(x[4]) * np.cos(x[4]) + b
    sigma_22 = x[2] * np.sin(x[4]) ** 2 + x[3] * np.cos(x[4]) ** 2 + c

    # Determinant of the sum of covariances
    det_sigma = np.abs(sigma_11 * sigma_22 - sigma_12 ** 2)
    x_inv_sigma_y_numerator = (
        sigma_22 * mu_1 ** 2 - 2 * sigma_12 * mu_1 * mu_2 + sigma_11 * mu_2 ** 2
    )

    if det_sigma < 1e-32:
        return (
            mu_1 ** 2 + mu_2 ** 2,
            np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        )

    dist = x_inv_sigma_y_numerator / det_sigma + np.log(det_sigma) + np.log(2 * np.pi)

    grad = np.zeros(5, np.float32)
    grad[0] = (2 * sigma_22 * mu_1 - 2 * sigma_12 * mu_2) / det_sigma
    grad[1] = (2 * sigma_11 * mu_2 - 2 * sigma_12 * mu_1) / det_sigma

    grad[2] = mu_2 * (mu_2 * np.cos(x[4]) ** 2 - mu_1 * np.cos(x[4]) * np.sin(x[4]))
    grad[2] += mu_1 * (mu_1 * np.sin(x[4]) ** 2 - mu_2 * np.cos(x[4]) * np.sin(x[4]))
    grad[2] *= det_sigma
    grad[2] -= x_inv_sigma_y_numerator * np.cos(x[4]) ** 2 * sigma_22
    grad[2] -= x_inv_sigma_y_numerator * np.sin(x[4]) ** 2 * sigma_11
    grad[2] += x_inv_sigma_y_numerator * 2 * sigma_12 * np.sin(x[4]) * np.cos(x[4])
    grad[2] /= det_sigma ** 2 + 1e-8

    grad[3] = mu_1 * (mu_1 * np.cos(x[4]) ** 2 - mu_2 * np.cos(x[4]) * np.sin(x[4]))
    grad[3] += mu_2 * (mu_2 * np.sin(x[4]) ** 2 - mu_1 * np.cos(x[4]) * np.sin(x[4]))
    grad[3] *= det_sigma
    grad[3] -= x_inv_sigma_y_numerator * np.sin(x[4]) ** 2 * sigma_22
    grad[3] -= x_inv_sigma_y_numerator * np.cos(x[4]) ** 2 * sigma_11
    grad[3] -= x_inv_sigma_y_numerator * 2 * sigma_12 * np.sin(x[4]) * np.cos(x[4])
    grad[3] /= det_sigma ** 2 + 1e-8

    grad[4] = (x[3] - x[2]) * (
        2 * mu_1 * mu_2 * np.cos(2 * x[4]) - (mu_1 ** 2 - mu_2 ** 2) * np.sin(2 * x[4])
    )
    grad[4] *= det_sigma
    grad[4] -= x_inv_sigma_y_numerator * (x[3] - x[2]) * np.sin(2 * x[4]) * sigma_22
    grad[4] -= x_inv_sigma_y_numerator * (x[2] - x[3]) * np.sin(2 * x[4]) * sigma_11
    grad[4] -= x_inv_sigma_y_numerator * 2 * sigma_12 * (x[2] - x[3]) * np.cos(2 * x[4])
    grad[4] /= det_sigma ** 2 + 1e-8

    return dist, grad


@numba.njit(fastmath=True)
def spherical_gaussian_grad(x, y):
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = x[2] + y[2]
    sigma_sign = np.sign(sigma)

    if sigma == 0:
        return 10.0, np.array([0.0, 0.0, -1.0], dtype=np.float32)

    dist = (
        (mu_1 ** 2 + mu_2 ** 2) / np.abs(sigma)
        + 2 * np.log(np.abs(sigma))
        + np.log(2 * np.pi)
    )
    grad = np.empty(3, dtype=np.float32)

    grad[0] = (2 * mu_1) / np.abs(sigma)
    grad[1] = (2 * mu_2) / np.abs(sigma)
    grad[2] = sigma_sign * (
        -(mu_1 ** 2 + mu_2 ** 2) / (sigma ** 2) + (2 / np.abs(sigma))
    )

    return dist, grad


# Special discrete distances -- where x and y are objects, not vectors


def get_discrete_params(data, metric):
    if metric == "ordinal":
        return {"support_size": float(data.max() - data.min()) / 2.0}
    elif metric == "count":
        min_count = scipy.stats.tmin(data)
        max_count = scipy.stats.tmax(data)
        lambda_ = scipy.stats.tmean(data)
        normalisation = count_distance(min_count, max_count, poisson_lambda=lambda_)
        return {
            "poisson_lambda": lambda_,
            "normalisation": normalisation / 2.0,  # heuristic
        }
    elif metric == "string":
        lengths = np.array([len(x) for x in data])
        max_length = scipy.stats.tmax(lengths)
        max_dist = max_length / 1.5  # heuristic
        normalisation = max_dist / 2.0  # heuristic
        return {"normalisation": normalisation, "max_dist": max_dist / 2.0}  # heuristic

    else:
        return {}


@numba.jit()
def categorical_distance(x, y):
    if x == y:
        return 0.0
    else:
        return 1.0


@numba.jit()
def hierarchical_categorical_distance(x, y, cat_hierarchy=[{}]):
    n_levels = float(len(cat_hierarchy))
    for level, cats in enumerate(cat_hierarchy):
        if cats[x] == cats[y]:
            return float(level) / n_levels
    else:
        return 1.0


@numba.njit()
def ordinal_distance(x, y, support_size=1.0):
    return abs(x - y) / support_size


@numba.jit()
def count_distance(x, y, poisson_lambda=1.0, normalisation=1.0):
    lo = int(min(x, y))
    hi = int(max(x, y))

    log_lambda = np.log(poisson_lambda)

    if lo < 2:
        log_k_factorial = 0.0
    elif lo < 10:
        log_k_factorial = 0.0
        for k in range(2, lo):
            log_k_factorial += np.log(k)
    else:
        log_k_factorial = approx_log_Gamma(lo + 1)

    result = 0.0

    for k in range(lo, hi):
        result += k * log_lambda - poisson_lambda - log_k_factorial
        log_k_factorial += np.log(k)

    return result / normalisation


@numba.njit()
def levenshtein(x, y, normalisation=1.0, max_distance=20):
    x_len, y_len = len(x), len(y)

    # Opt out of some comparisons
    if abs(x_len - y_len) > max_distance:
        return abs(x_len - y_len) / normalisation

    v0 = np.arange(y_len + 1).astype(np.float64)
    v1 = np.zeros(y_len + 1)

    for i in range(x_len):

        v1[i] = i + 1

        for j in range(y_len):
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1[j] + 1
            substitution_cost = int(x[i] == y[j])

            v1[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)

        v0 = v1

        # Abort early if we've already exceeded max_dist
        if np.min(v0) > max_distance:
            return max_distance / normalisation

    return v0[y_len] / normalisation


named_distances = {
    # general minkowski distances
    "euclidean": euclidean,
    "l2": euclidean,
    "manhattan": manhattan,
    "taxicab": manhattan,
    "l1": manhattan,
    "chebyshev": chebyshev,
    "linfinity": chebyshev,
    "linfty": chebyshev,
    "linf": chebyshev,
    "minkowski": minkowski,
    "poincare": poincare,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean,
    "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski,
    "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis,
    # Other distances
    "canberra": canberra,
    "cosine": cosine,
    "correlation": correlation,
    "hellinger": hellinger,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    "ll_dirichlet": ll_dirichlet,
    "symmetric_kl": symmetric_kl,
    # Binary distances
    "hamming": hamming,
    "jaccard": jaccard,
    "dice": dice,
    "matching": matching,
    "kulsinski": kulsinski,
    "rogerstanimoto": rogers_tanimoto,
    "russellrao": russellrao,
    "sokalsneath": sokal_sneath,
    "sokalmichener": sokal_michener,
    "yule": yule,
    # Special discrete distances
    "categorical": categorical_distance,
    "ordinal": ordinal_distance,
    "hierarchical_categorical": hierarchical_categorical_distance,
    "count": count_distance,
    "string": levenshtein,
}

named_distances_with_gradients = {
    # general minkowski distances
    "euclidean": euclidean_grad,
    "l2": euclidean_grad,
    "manhattan": manhattan_grad,
    "taxicab": manhattan_grad,
    "l1": manhattan_grad,
    "chebyshev": chebyshev_grad,
    "linfinity": chebyshev_grad,
    "linfty": chebyshev_grad,
    "linf": chebyshev_grad,
    "minkowski": minkowski_grad,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean_grad,
    "standardised_euclidean": standardised_euclidean_grad,
    "wminkowski": weighted_minkowski_grad,
    "weighted_minkowski": weighted_minkowski_grad,
    "mahalanobis": mahalanobis_grad,
    # Other distances
    "canberra": canberra_grad,
    "cosine": cosine_grad,
    "correlation": correlation_grad,
    "hellinger": hellinger_grad,
    "haversine": haversine_grad,
    "braycurtis": bray_curtis_grad,
    "symmetric_kl": symmetric_kl_grad,
    # Special embeddings
    "spherical_gaussian_energy": spherical_gaussian_energy_grad,
    "diagonal_gaussian_energy": diagonal_gaussian_energy_grad,
    "gaussian_energy": gaussian_energy_grad,
    "hyperboloid": hyperboloid_grad,
}

DISCRETE_METRICS = (
    "categorical",
    "hierarchical_categorical",
    "ordinal",
    "count",
    "string",
)

SPECIAL_METRICS = (
    "hellinger",
    "ll_dirichlet",
    "symmetric_kl",
    "poincare",
    hellinger,
    ll_dirichlet,
    symmetric_kl,
    poincare,
)


@numba.njit(parallel=True)
def parallel_special_metric(X, Y=None, metric=hellinger):
    if Y is None:
        result = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                result[i, j] = metric(X[i], X[j])
                result[j, i] = result[i, j]
    else:
        result = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                result[i, j] = metric(X[i], Y[j])

    return result


def pairwise_special_metric(X, Y=None, metric="hellinger", kwds=None):
    if callable(metric):
        if kwds is not None:
            kwd_vals = tuple(kwds.values())
        else:
            kwd_vals = ()

        @numba.njit(fastmath=True)
        def _partial_metric(_X, _Y=None):
            return metric(_X, _Y, *kwd_vals)

        return pairwise_distances(X, Y, metric=_partial_metric)
    else:
        special_metric_func = named_distances[metric]
    return parallel_special_metric(X, Y, metric=special_metric_func)


#jumap_#############

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def breadth_first_search(adjmat, start, min_vertices):
    explored = []
    queue = [start]
    levels = {}
    levels[start] = 0
    max_level = np.inf
    visited = [start]

    while queue:
        node = queue.pop(0)
        explored.append(node)
        if max_level == np.inf and len(explored) > min_vertices:
            max_level = max(levels.values())

        if levels[node] + 1 < max_level:
            neighbors = adjmat[node].indices
            for neighbour in neighbors:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

                    levels[neighbour] = levels[node] + 1

    return np.array(explored)


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
    k: float
        The number of nearest neighbors to approximate for.
    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.
    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.
    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    low_memory=False,
    use_pynndescent=True,
    verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.
    metric: string or callable
        The metric to use for the computation.
    metric_kwds: dict
        Any arguments to pass to the metric computation function.
    angular: bool
        Whether to use angular rp trees in NN approximation.
    random_state: np.random state
        The random state to use for approximate NN computations.
    low_memory: bool (optional, default False)
        Whether to pursue lower memory NNdescent.
    verbose: bool (optional, default False)
        Whether to print status data during the computation.
    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise)
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = fast_knn_indices(X, n_neighbors)
        # knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        # TODO: Hacked values for now
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        if _HAVE_PYNNDESCENT and use_pynndescent:
            nnd = NNDescent(
                X,
                n_neighbors=n_neighbors,
                metric=metric,
                metric_kwds=metric_kwds,
                random_state=random_state,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                low_memory=low_memory,
                verbose=verbose,
            )
            knn_indices, knn_dists = nnd.neighbor_graph
            rp_forest = nnd
        else:
            # Otherwise fall back to nn descent in Jvis
            if callable(metric):
                _distance_func = metric
            elif metric in named_distances:
                _distance_func = named_distances[metric]
            else:
                raise ValueError("Metric is neither callable, nor a recognised string")

            rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

            if scipy.sparse.isspmatrix_csr(X):
                if callable(metric):
                    _distance_func = metric
                else:
                    try:
                        _distance_func = sparse_named_distances[metric]
                        if metric in sparse_need_n_features:
                            metric_kwds["n_features"] = X.shape[1]
                    except KeyError as e:
                        raise ValueError(
                            "Metric {} not supported for sparse data".format(metric)
                        ) from e

                # Create a partial function for distances with arguments
                if len(metric_kwds) > 0:
                    dist_args = tuple(metric_kwds.values())

                    @numba.njit()
                    def _partial_dist_func(ind1, data1, ind2, data2):
                        return _distance_func(ind1, data1, ind2, data2, *dist_args)

                    distance_func = _partial_dist_func
                else:
                    distance_func = _distance_func
                # metric_nn_descent = make_sparse_nn_descent(
                #     distance_func, tuple(metric_kwds.values())
                # )

                if verbose:
                    print(ts(), "Building RP forest with", str(n_trees), "trees")

                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
                leaf_array = rptree_leaf_array(rp_forest)

                if verbose:
                    print(ts(), "NN descent for", str(n_iters), "iterations")
                knn_indices, knn_dists = sparse_nn_descent(
                    X.indices,
                    X.indptr,
                    X.data,
                    X.shape[0],
                    n_neighbors,
                    rng_state,
                    max_candidates=60,
                    sparse_dist=distance_func,
                    low_memory=low_memory,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    n_iters=n_iters,
                    verbose=verbose,
                )
            else:
                # metric_nn_descent = make_nn_descent(
                #     distance_func, tuple(metric_kwds.values())
                # )
                if len(metric_kwds) > 0:
                    dist_args = tuple(metric_kwds.values())

                    @numba.njit()
                    def _partial_dist_func(x, y):
                        return _distance_func(x, y, *dist_args)

                    distance_func = _partial_dist_func
                else:
                    distance_func = _distance_func

                if verbose:
                    print(ts(), "Building RP forest with", str(n_trees), "trees")
                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
                leaf_array = rptree_leaf_array(rp_forest)
                if verbose:
                    print(ts(), "NN descent for", str(n_iters), "iterations")
                knn_indices, knn_dists = nn_descent(
                    X,
                    n_neighbors,
                    rng_state,
                    max_candidates=60,
                    dist=distance_func,
                    low_memory=low_memory,
                    rp_tree_init=True,
                    leaf_array=leaf_array,
                    n_iters=n_iters,
                    verbose=verbose,
                )

            if np.any(knn_indices < 0):
                warn(
                    "Failed to correctly find n_neighbors for some samples."
                    "Results may be less than ideal. Try re-running with"
                    "different parameters."
                )
    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, rp_forest


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)

def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    rhos: array of shape(n_samples)
        The local connectivity adjustment.
    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)
    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)
    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * ll_dirichlet
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.
    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.
    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.
    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose
        )

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    return result, sigmas, rhos


@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.
    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.
    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.
    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.
    target: array of shape (n_samples)
        The categorical labels to use in the intersection.
    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.
    far_dist float (optional, default 5.0)
        The distance between unmatched labels.
    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return


@numba.jit()
def fast_metric_intersection(
    rows, cols, values, discrete_space, metric, metric_args, scale
):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.
    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.
    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.
    values: array of shape
        An array of the values of each non-zero in the sparse matrix
        representation.
    discrete_space: array of shape (n_samples, n_features)
        The vectors of categorical labels to use in the intersection.
    metric: numba function
        The function used to calculate distance over the target array.
    scale: float
        A scaling to apply to the metric.
    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        dist = metric(discrete_space[i], discrete_space[j], *metric_args)
        values[nz] *= np.exp(-(scale * dist))

    return


@numba.njit()
def reprocess_row(probabilities, k=15, n_iters=32):
    target = np.log2(k)

    lo = 0.0
    hi = NPY_INFINITY
    mid = 1.0

    for n in range(n_iters):

        psum = 0.0
        for j in range(probabilities.shape[0]):
            psum += pow(probabilities[j], mid)

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum < target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    return np.power(probabilities, mid)


@numba.njit()
def reset_local_metrics(simplicial_set_indptr, simplicial_set_data):
    for i in range(simplicial_set_indptr.shape[0] - 1):
        simplicial_set_data[
            simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]
        ] = reprocess_row(
            simplicial_set_data[simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]]
        )
    return


def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.
    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.
    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        simplicial_set = simplicial_set.tocsr()
        reset_local_metrics(simplicial_set.indptr, simplicial_set.data)
        simplicial_set = simplicial_set.tocoo()
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


def discrete_metric_simplicial_set_intersection(
    simplicial_set,
    discrete_space,
    unknown_dist=1.0,
    far_dist=5.0,
    metric=None,
    metric_kws={},
    metric_scale=1.0,
):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from discrete metric data using discrete distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.
    TODO: optional category cardinality based weighting of distance
    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.
    discrete_space: array of shape (n_samples)
        The categorical labels to use in the intersection.
    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.
    far_dist: float (optional, default 5.0)
        The distance between unmatched labels.
    metric: str (optional, default None)
        If not None, then use this metric to determine the
        distance between values.
    metric_scale: float (optional, default 1.0)
        If using a custom metric scale the distance values by
        this value -- this controls the weighting of the
        intersection. Larger values weight more toward target.
    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    if metric is not None:
        # We presume target is now a 2d array, with each row being a
        # vector of target info
        if metric in named_distances:
            metric_func = named_distances[metric]
        else:
            raise ValueError("Discrete intersection metric is not recognized")

        fast_metric_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            metric_func,
            tuple(metric_kws.values()),
            metric_scale,
        )
    else:
        fast_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            unknown_dist,
            far_dist,
        )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight):

    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        weight,
    )

    return result


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    output_metric=named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=False,
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            output_metric,
            tuple(output_metric_kwds.values()),
            verbose=verbose,
        )

    return embedding


@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).
    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample
    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.
    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.
    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAP(BaseEstimator):
    """Uniform Manifold Approximation and Projection
    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.
    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
    low_memory: bool (optional, default False)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that UMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.
    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.
    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.
    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.
    target_metric_kwds: dict (optional, default None)
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.
    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.
    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.
    verbose: bool (optional, default False)
        Controls verbosity of logging.
    unique: bool (optional, default False)
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        verbose=False,
        unique=False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer of at least 10")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method method, define a distance function that returns "
                    "a tuple of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn(
                "using precomputed metric; transform will be unavailable for new data and inverse_transform "
                "will be unavailable for all data"
            )
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in named_distances:
            if self._sparse_data:
                if self.metric in sparse_named_distances:
                    self._input_distance_func = sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = named_distances[self.metric]
            try:
                self._inverse_distance_func = named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set ooutput distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in named_distances_with_gradients:
            self._output_distance_func = named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))

        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    def fit(self, X, X2 = None, alpha = 1.0, y=None):
        """Fit X into an embedded space.
        Optionally use y for supervised dimension reduction.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        # Check if we should unique the data
        # We've already ensured that we aren't in the precomputed case
        if self.unique:
            # check if the matrix is dense
            if self._sparse_data:
                # Call a sparse unique function
                index, inverse, counts = csr_unique(X)
            else:
                index, inverse, counts = np.unique(
                    X,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    axis=0,
                )[1:4]
            if self.verbose:
                print(
                    "Unique=True -> Number of data points reduced from ",
                    X.shape[0],
                    " to ",
                    X[index].shape[0],
                )
                most_common = np.argmax(counts)
                print(
                    "Most common duplicate is",
                    index[most_common],
                    " with a count of ",
                    counts[most_common],
                )
        # If we aren't asking for unique use the full index.
        # This will save special cases later.
        else:
            index = list(range(X.shape[0]))
            inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        # Note: unless it causes issues for setting 'index', could move this to
        # initial sparsity check above
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        if self.metric == "precomputed" and self._sparse_data:
            # For sparse precomputed distance matrices, we just argsort the rows to find
            # nearest neighbors. To make this easier, we expect matrices that are
            # symmetrical (so we can find neighbors by looking at rows in isolation,
            # rather than also having to consider that sample's column too).
            print("Computing KNNs for sparse precomputed distances...")
            if sparse_tril(X).getnnz() != sparse_triu(X).getnnz():
                raise ValueError(
                    "Sparse precomputed distance matrices should be symmetrical!"
                )
            if not np.all(X.diagonal() == 0):
                raise ValueError("Non-zero distances from samples to themselves!")
            self._knn_indices = np.zeros((X.shape[0], self.n_neighbors), dtype=np.int)
            self._knn_dists = np.zeros(self._knn_indices.shape, dtype=np.float)
            for row_id in range(X.shape[0]):
                # Find KNNs row-by-row
                row_data = X[row_id].data
                row_indices = X[row_id].indices
                if len(row_data) < self._n_neighbors:
                    raise ValueError(
                        "Some rows contain fewer than n_neighbors distances!"
                    )
                row_nn_data_indices = np.argsort(row_data)[: self._n_neighbors]
                self._knn_indices[row_id] = row_indices[row_nn_data_indices]
                self._knn_dists[row_id] = row_data[row_nn_data_indices]
            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
            )
        # Handle small cases efficiently by computing all distances
        # elif X[index].shape[0] < 4096 and not self.force_approximation_algorithm:
        elif X[index].shape[0] < 14096 and not self.force_approximation_algorithm: ## My code
            # print("Always use exact mode")
            self._small_data = True
            try:
                # sklearn pairwise_distances fails for callable metric on sparse data
                _m = self.metric if self._sparse_data else self._input_distance_func
                dmat = pairwise_distances(X[index], metric=_m, **self._metric_kwds)
            except (ValueError, TypeError) as e:
                # metric is numba.jit'd or not supported by sklearn,
                # fallback to pairwise special

                if self._sparse_data:
                    # Get a fresh metric since we are casting to dense
                    if not callable(self.metric):
                        _m = named_distances[self.metric]
                        dmat = pairwise_special_metric(
                            X[index].toarray(), metric=_m, kwds=self._metric_kwds,
                        )
                    else:
                        dmat = pairwise_special_metric(
                            X[index],
                            metric=self._input_distance_func,
                            kwds=self._metric_kwds,
                        )
                else:
                    dmat = pairwise_special_metric(
                        X[index],
                        metric=self._input_distance_func,
                        kwds=self._metric_kwds,
                    )
            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
                )

            ## Compute for second modality
            if alpha < 0.99:
                # print("Compute jointUMAP")
                try:
                    # sklearn pairwise_distances fails for callable metric on sparse data
                    _m2 = self.metric if self._sparse_data else self._input_distance_func
                    dmat2 = pairwise_distances(X2[index], metric=_m2, **self._metric_kwds)
                except (ValueError, TypeError) as e:
                    # metric is numba.jit'd or not supported by sklearn,
                    # fallback to pairwise special

                    if self._sparse_data:
                        # Get a fresh metric since we are casting to dense
                        if not callable(self.metric):
                            _m2 = named_distances[self.metric]
                            dmat2 = pairwise_special_metric(
                                X2[index].toarray(), metric=_m2, kwds=self._metric_kwds,
                            )
                        else:
                            dmat2 = pairwise_special_metric(
                                X2[index],
                                metric=self._input_distance_func,
                                kwds=self._metric_kwds,
                            )
                    else:
                        dmat2 = pairwise_special_metric(
                            X2[index],
                            metric=self._input_distance_func,
                            kwds=self._metric_kwds,
                        )
                self.graph2_, self._sigmas2, self._rhos2 = fuzzy_simplicial_set(
                    dmat2,
                    self._n_neighbors,
                    random_state,
                    "precomputed",
                    self._metric_kwds,
                    None,
                    None,
                    self.angular_rp_forest,
                    self.set_op_mix_ratio,
                    self.local_connectivity,
                    True,
                    self.verbose,
                    )
                self.graph_ = (self.graph_ + self.graph2_)/2.0
                self._sigmas = (self._sigmas + self._sigmas2)/2.0 
                self._rhos = (self._rhos + self._rhos2)/2.0
        else:
            # Standard case
            self._small_data = False
            # pass string identifier if pynndescent also defines distance metric
            if _HAVE_PYNNDESCENT:
                if self._sparse_data and self.metric in pynn_sparse_named_distances:
                    nn_metric = self.metric
                elif not self._sparse_data and self.metric in pynn_named_distances:
                    nn_metric = self.metric
                else:
                    nn_metric = self._input_distance_func
            else:
                nn_metric = self._input_distance_func
            (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                X[index],
                self._n_neighbors,
                nn_metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.low_memory,
                use_pynndescent=True,
                verbose=self.verbose,
            )

            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                nn_metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
            )

            if not _HAVE_PYNNDESCENT:
                self._search_graph = scipy.sparse.lil_matrix(
                    (X[index].shape[0], X[index].shape[0]), dtype=np.int8
                )
                _rows = []
                _data = []
                for i in self._knn_indices:
                    _non_neg = i[i >= 0]
                    _rows.append(_non_neg.tolist())
                    _data.append(np.ones(_non_neg.shape[0], dtype=np.int8).tolist())

                self._search_graph.rows = np.empty(len(_rows), dtype=object)
                self._search_graph.rows[:] = _rows
                self._search_graph.data = np.empty(len(_data), dtype=object)
                self._search_graph.data[:] = _data
                self._search_graph = self._search_graph.maximum(
                    self._search_graph.transpose()
                ).tocsr()

                if (self.metric != "precomputed") and (len(self._metric_kwds) > 0):
                    # Create a partial function for distances with arguments
                    _distance_func = self._input_distance_func
                    _dist_args = tuple(self._metric_kwds.values())
                    if self._sparse_data:

                        @numba.njit()
                        def _partial_dist_func(ind1, data1, ind2, data2):
                            return _distance_func(ind1, data1, ind2, data2, *_dist_args)

                        self._input_distance_func = _partial_dist_func
                    else:

                        @numba.njit()
                        def _partial_dist_func(x, y):
                            return _distance_func(x, y, *_dist_args)

                        self._input_distance_func = _partial_dist_func

        # Currently not checking if any duplicate points have differing labels
        # Might be worth throwing a warning...
        if y is not None:
            len_X = len(X) if not self._sparse_data else X.shape[0]
            if len_X != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len_X, len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)[index]
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            elif self.target_metric in DISCRETE_METRICS:
                if self.target_weight < 1.0:
                    scale = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    scale = 1.0e12
                # self.graph_ = discrete_metric_simplicial_set_intersection(
                #     self.graph_,
                #     y_,
                #     metric=self.target_metric,
                #     metric_kws=self.target_metric_kwds,
                #     metric_scale=scale
                # )

                metric_kws = get_discrete_params(y_, self.target_metric)

                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_,
                    y_,
                    metric=self.target_metric,
                    metric_kws=metric_kws,
                    metric_scale=scale,
                )
            else:
                if len(y_.shape) == 1:
                    y_ = y_.reshape(-1, 1)
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    try:
                        ydmat = pairwise_distances(
                            y_, metric=self.target_metric, **self._target_metric_kwds
                        )
                    except (TypeError, ValueError):
                        ydmat = pairwise_special_metric(
                            y_,
                            metric=self.target_metric,
                            kwds=self._target_metric_kwds,
                        )

                    target_graph, target_sigmas, target_rhos = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                else:
                    # Standard case
                    target_graph, target_sigmas, target_rhos = fuzzy_simplicial_set(
                        y_,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                # product = self.graph_.multiply(target_graph)
                # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
                # #                                        target_graph -
                # #                                        product)
                # self.graph_ = product
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight
                )
                self.graph_ = reset_local_connectivity(self.graph_)

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct embedding")

        self.embedding_ = simplicial_set_embedding(
            self._raw_data[index],  # JH why raw data?
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
        )[inverse]

        if self.verbose:
            print(ts() + " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

        return self

    def fit_transform(self, X, X2 = None, alpha = 1.0, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        #print("My name is Hoan")
        self.fit(X, X2, alpha, y)
        return self.embedding_

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        # If we fit just a single instance then error
        if self.embedding_.shape[0] == 1:
            raise ValueError(
                "Transform unavailable when model was fit with only a single data sample."
            )
        # If we just have the original input then short circuit things
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        x_hash = joblib.hash(X)
        if x_hash == self._input_hash:
            return self.embedding_

        if self.metric == "precomputed":
            raise ValueError(
                "Transform  of new data not available for precomputed metric."
            )

        # X = check_array(X, dtype=np.float32, order="C", accept_sparse="csr")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if self._small_data:
            try:
                # sklearn pairwise_distances fails for callable metric on sparse data
                _m = self.metric if self._sparse_data else self._input_distance_func
                dmat = pairwise_distances(
                    X, self._raw_data, metric=_m, **self._metric_kwds
                )
            except (TypeError, ValueError):
                dmat = pairwise_special_metric(
                    X,
                    self._raw_data,
                    metric=self._input_distance_func,
                    kwds=self._metric_kwds,
                )
            indices = np.argpartition(dmat, self._n_neighbors)[:, : self._n_neighbors]
            dmat_shortened = submatrix(dmat, indices, self._n_neighbors)
            indices_sorted = np.argsort(dmat_shortened)
            indices = submatrix(indices, indices_sorted, self._n_neighbors)
            dists = submatrix(dmat_shortened, indices_sorted, self._n_neighbors)
        elif _HAVE_PYNNDESCENT:
            indices, dists = self._rp_forest.query(X, self.n_neighbors)
        elif self._sparse_data:
            if not scipy.sparse.issparse(X):
                X = scipy.sparse.csr_matrix(X)

            init = sparse_initialise_search(
                self._rp_forest,
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                X.indices,
                X.indptr,
                X.data,
                int(
                    self._n_neighbors
                    * self.transform_queue_size
                    * (1 + int(self._sparse_data))
                ),
                rng_state,
                self._input_distance_func,
            )
            result = sparse_initialized_nnd_search(
                self._raw_data.indices,
                self._raw_data.indptr,
                self._raw_data.data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                X.indices,
                X.indptr,
                X.data,
                self._input_distance_func,
            )

            indices, dists = deheap_sort(result)
            indices = indices[:, : self._n_neighbors]
            dists = dists[:, : self._n_neighbors]
        else:
            init = initialise_search(
                self._rp_forest,
                self._raw_data,
                X,
                int(self._n_neighbors * self.transform_queue_size),
                rng_state,
                self._input_distance_func,
            )
            result = initialized_nnd_search(
                self._raw_data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                X,
                self._input_distance_func,
            )

            indices, dists = deheap_sort(result)
            indices = indices[:, : self._n_neighbors]
            dists = dists[:, : self._n_neighbors]

        dists = dists.astype(np.float32, order="C")

        adjusted_local_connectivity = max(0.0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist(
            dists,
            float(self._n_neighbors),
            local_connectivity=float(adjusted_local_connectivity),
        )

        rows, cols, vals = compute_membership_strengths(indices, dists, sigmas, rhos)

        graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # This was a very specially constructed graph with constant degree.
        # That lets us do fancy unpacking by reshaping the csr matrix indices
        # and data. Doing so relies on the constant degree assumption!
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
        weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
        embedding = init_transform(inds, weights, self.embedding_)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = int(self.n_epochs // 3.0)

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        # optimize_layout = make_optimize_layout(
        #     self._output_distance_func,
        #     tuple(self.output_metric_kwds.values()),
        # )

        if self.output_metric == "euclidean":
            embedding = optimize_layout_euclidean(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  # Fixes #179 & #217,
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self.random_state is None,
                verbose=self.verbose,
            )
        else:
            embedding = optimize_layout_generic(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  # Fixes #179 & #217
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self._output_distance_func,
                tuple(self._output_metric_kwds.values()),
                verbose=self.verbose,
            )

        return embedding

    def inverse_transform(self, X):
        """Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """

        if self._sparse_data:
            raise ValueError("Inverse transform not available for sparse input.")
        elif self._inverse_distance_func is None:
            raise ValueError("Inverse transform not available for given metric.")
        elif self.n_components >= 8:
            warn(
                "Inverse transform works best with low dimensional embeddings."
                " Results may be poor, or this approach to inverse transform"
                " may fail altogether! If you need a high dimensional latent"
                " space and inverse transform operations consider using an"
                " autoencoder."
            )

        X = check_array(X, dtype=np.float32, order="C")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        # build Delaunay complex (Does this not assume a roughly euclidean output metric)?
        deltri = scipy.spatial.Delaunay(
            self.embedding_, incremental=True, qhull_options="QJ"
        )
        neighbors = deltri.simplices[deltri.find_simplex(X)]
        adjmat = scipy.sparse.lil_matrix(
            (self.embedding_.shape[0], self.embedding_.shape[0]), dtype=int
        )
        for i in np.arange(0, deltri.simplices.shape[0]):
            for j in deltri.simplices[i]:
                if j < self.embedding_.shape[0]:
                    idx = deltri.simplices[i][
                        deltri.simplices[i] < self.embedding_.shape[0]
                    ]
                    adjmat[j, idx] = 1
                    adjmat[idx, j] = 1

        adjmat = scipy.sparse.csr_matrix(adjmat)

        min_vertices = min(self._raw_data.shape[-1], self._raw_data.shape[0])

        neighborhood = [
            breadth_first_search(adjmat, v[0], min_vertices=min_vertices)
            for v in neighbors
        ]
        if callable(self.output_metric):
            # need to create another numba.jit-able wrapper for callable
            # output_metrics that return a tuple (already checked that it does
            # during param validation in `fit` method)
            _out_m = self.output_metric

            @numba.njit(fastmath=True)
            def _output_dist_only(x, y, *kwds):
                return _out_m(x, y, *kwds)[0]

            dist_only_func = _output_dist_only
        elif self.output_metric in named_distances.keys():
            dist_only_func = named_distances[self.output_metric]
        else:
            # shouldn't really ever get here because of checks already performed,
            # but works as a failsafe in case attr was altered manually after fitting
            raise ValueError(
                "Unrecognized output metric: {}".format(self.output_metric)
            )

        dist_args = tuple(self._output_metric_kwds.values())
        distances = [
            np.array(
                [
                    dist_only_func(X[i], self.embedding_[nb], *dist_args)
                    for nb in neighborhood[i]
                ]
            )
            for i in range(X.shape[0])
        ]
        idx = np.array([np.argsort(e)[:min_vertices] for e in distances])

        dists_output_space = np.array(
            [distances[i][idx[i]] for i in range(len(distances))]
        )
        indices = np.array([neighborhood[i][idx[i]] for i in range(len(neighborhood))])

        rows, cols, distances = np.array(
            [
                [i, indices[i, j], dists_output_space[i, j]]
                for i in range(indices.shape[0])
                for j in range(min_vertices)
            ]
        ).T

        # calculate membership strength of each edge
        weights = 1 / (1 + self._a * distances ** (2 * self._b))

        # compute 1-skeleton
        # convert 1-skeleton into coo_matrix adjacency matrix
        graph = scipy.sparse.coo_matrix(
            (weights, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # That lets us do fancy unpacking by reshaping the csr matrix indices
        # and data. Doing so relies on the constant degree assumption!
        # csr_graph = graph.tocsr()
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.indices.reshape(X.shape[0], min_vertices)
        weights = csr_graph.data.reshape(X.shape[0], min_vertices)
        inv_transformed_points = init_transform(inds, weights, self._raw_data)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = int(self.n_epochs // 3.0)

        # graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        # graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        inv_transformed_points = optimize_layout_inverse(
            inv_transformed_points,
            self._raw_data,
            head,
            tail,
            weight,
            self._sigmas,
            self._rhos,
            n_epochs,
            graph.shape[1],
            epochs_per_sample,
            self._a,
            self._b,
            rng_state,
            self.repulsion_strength,
            self._initial_alpha / 4.0,
            self.negative_sample_rate,
            self._inverse_distance_func,
            tuple(self._metric_kwds.values()),
            verbose=self.verbose,
        )

        return inv_transformed_points


class DataFrameUMAP(BaseEstimator):
    def __init__(
        self,
        metrics,
        n_neighbors=15,
        n_components=2,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        verbose=False,
    ):
        self.metrics = metrics
        self.n_neighbors = n_neighbors
        self.output_metric = output_metric
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.verbose = verbose

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist must be greater than 0.0")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 2")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 2")
        if not isinstance(self.n_components, int):
            raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer " "larger than 10")
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds

        if callable(self.output_metric):
            self._output_distance_func = self.output_metric
        elif (
            self.output_metric in named_distances
            and self.output_metric in named_distances_with_gradients
        ):
            self._output_distance_func = named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        else:
            if self.output_metric in named_distances:
                raise ValueError(
                    "gradient function is not yet implemented for "
                    + repr(self.output_metric)
                    + "."
                )
            else:
                raise ValueError(
                    "output_metric is neither callable, " + "nor a recognised string"
                )

        # validate metrics argument
        assert isinstance(self.metrics, list) or self.metrics == "infer"
        if self.metrics != "infer":
            for item in self.metrics:
                assert isinstance(item, tuple) and len(item) == 3
                assert isinstance(item[0], str)
                assert item[1] in named_distances
                assert isinstance(item[2], list) and len(item[2]) >= 1

                for col in item[2]:
                    assert isinstance(col, str) or isinstance(col, int)

    def fit(self, X, y=None):

        self._validate_parameters()

        # X should be a pandas dataframe, or np.array; check
        # how column transformer handles this.
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        # Error check n_neighbors based on data size
        if X.shape[0] <= self.n_neighbors:
            if X.shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        if self.metrics == "infer":
            raise NotImplementedError("Metric inference not implemented yet")

        random_state = check_random_state(self.random_state)

        self.metric_graphs_ = {}
        self._sigmas = {}
        self._rhos = {}
        self._knn_indices = {}
        self._knn_dists = {}
        self._rp_forest = {}
        self.graph_ = None

        def is_discrete_metric(metric_data):
            return metric_data[1] in DISCRETE_METRICS

        for metric_data in sorted(self.metrics, key=is_discrete_metric):
            name, metric, columns = metric_data
            print(name, metric, columns)

            if metric in DISCRETE_METRICS:
                self.metric_graphs_[name] = None
                for col in columns:

                    discrete_space = X[col].values
                    metric_kws = get_discrete_params(discrete_space, metric)

                    self.graph_ = discrete_metric_simplicial_set_intersection(
                        self.graph_,
                        discrete_space,
                        metric=metric,
                        metric_kws=metric_kws,
                    )
            else:
                # Sparse not supported yet
                sub_data = check_array(
                    X[columns], dtype=np.float32, accept_sparse=False
                )

                if X.shape[0] < 4096:
                    # small case
                    self._small_data = True
                    # TODO: metric keywords not supported yet!
                    if metric in ("ll_dirichlet", "hellinger"):
                        dmat = pairwise_special_metric(sub_data, metric=metric)
                    else:
                        dmat = pairwise_distances(sub_data, metric=metric)

                    (
                        self.metric_graphs_[name],
                        self._sigmas[name],
                        self._rhos[name],
                    ) = fuzzy_simplicial_set(
                        dmat,
                        self._n_neighbors,
                        random_state,
                        "precomputed",
                        {},
                        None,
                        None,
                        self.angular_rp_forest,
                        self.set_op_mix_ratio,
                        self.local_connectivity,
                        False,
                        self.verbose,
                    )
                else:
                    self._small_data = False
                    # Standard case
                    # TODO: metric keywords not supported yet!
                    (
                        self._knn_indices[name],
                        self._knn_dists[name],
                        self._rp_forest[name],
                    ) = nearest_neighbors(
                        sub_data,
                        self._n_neighbors,
                        metric,
                        {},
                        self.angular_rp_forest,
                        random_state,
                        use_pynndescent=True,
                        verbose=self.verbose,
                    )

                    (
                        self.metric_graphs_[name],
                        self._sigmas[name],
                        self._rhos[name],
                    ) = fuzzy_simplicial_set(
                        sub_data,
                        self.n_neighbors,
                        random_state,
                        metric,
                        {},
                        self._knn_indices[name],
                        self._knn_dists[name],
                        self.angular_rp_forest,
                        self.set_op_mix_ratio,
                        self.local_connectivity,
                        False,
                        self.verbose,
                    )
                    # TODO: set up transform data

                if self.graph_ is None:
                    self.graph_ = self.metric_graphs_[name]
                else:
                    self.graph_ = general_simplicial_set_intersection(
                        self.graph_, self.metric_graphs_[name], 0.5
                    )

            print(self.graph_.data)
            self.graph_ = reset_local_connectivity(
                self.graph_, reset_local_metrics=True
            )

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print("Construct embedding")

        # TODO: Handle connected component issues properly
        # For now we just use manhattan and hope.
        self.embedding_ = simplicial_set_embedding(
            self._raw_data,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            "manhattan",
            {},
            self._output_distance_func,
            self.output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
        )

        self._input_hash = joblib.hash(self._raw_data)

        return self


class JUMAPBASE(BaseEstimator):
    """Uniform Manifold Approximation and Projection
    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.
    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
    low_memory: bool (optional, default False)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that UMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.
    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.
    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.
    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.
    target_metric_kwds: dict (optional, default None)
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.
    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.
    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.
    verbose: bool (optional, default False)
        Controls verbosity of logging.
    unique: bool (optional, default False)
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        verbose=False,
        unique=False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer of at least 10")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method method, define a distance function that returns "
                    "a tuple of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn(
                "using precomputed metric; transform will be unavailable for new data and inverse_transform "
                "will be unavailable for all data"
            )
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in named_distances:
            if self._sparse_data:
                if self.metric in sparse_named_distances:
                    self._input_distance_func = sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = named_distances[self.metric]
            try:
                self._inverse_distance_func = named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set ooutput distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in named_distances_with_gradients:
            self._output_distance_func = named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))

        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    def fit(self, jointX, alpha):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """

        X = next(iter(jointX.values()))
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        index = list(range(X.shape[0]))
        inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        # Note: unless it causes issues for setting 'index', could move this to
        # initial sparsity check above
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        if self.metric == "precomputed" and self._sparse_data:
            print("Not implemented")
        # elif X[index].shape[0] < 14096 and not self.force_approximation_algorithm: ## My code
        elif X[index].shape[0] < 13096 and not self.force_approximation_algorithm: ## My code
            # print("Exact mode: JUMAP")
            self._small_data = True
            self.graph_vec = [None]*len(jointX)
            self.sigma_vec = [None]*len(jointX)
            self.rho_vec = [None]*len(jointX)

            for idx, key in enumerate(jointX):
                try:
                    # sklearn pairwise_distances fails for callable metric on sparse data
                    _m = self.metric if self._sparse_data else self._input_distance_func
                    dmat = pairwise_distances(jointX[key][index], metric=_m, **self._metric_kwds)
                except (ValueError, TypeError) as e:
                    # metric is numba.jit'd or not supported by sklearn,
                    # fallback to pairwise special

                    if self._sparse_data:
                        # Get a fresh metric since we are casting to dense
                        if not callable(self.metric):
                            _m = named_distances[self.metric]
                            dmat = pairwise_special_metric(
                                jointX[key][index].toarray(), metric=_m, kwds=self._metric_kwds,
                            )
                        else:
                            dmat = pairwise_special_metric(
                                jointX[key][index],
                                metric=self._input_distance_func,
                                kwds=self._metric_kwds,
                            )
                    else:
                        dmat = pairwise_special_metric(
                            jointX[key][index],
                            metric=self._input_distance_func,
                            kwds=self._metric_kwds,
                        )
                self.graph_vec[idx], self.sigma_vec[idx], self.rho_vec[idx] = fuzzy_simplicial_set(
                    dmat,
                    self._n_neighbors,
                    random_state,
                    "precomputed",
                    self._metric_kwds,
                    None,
                    None,
                    self.angular_rp_forest,
                    self.set_op_mix_ratio,
                    self.local_connectivity,
                    True,
                    self.verbose,
                    )
            
            self.graph_  = alpha[0] * self.graph_vec[0]
            self._sigmas = alpha[0] * self.sigma_vec[0]
            self._rhos   = alpha[0] * self.rho_vec[0]
            # self._sigmas = np.average(np.array(self.sigma_vec))
            # self._rhos = np.average(np.array(self.rho_vec))
            for it in range(1, len(jointX)):
                self.graph_ += alpha[it] * self.graph_vec[it]
                self._sigmas += alpha[it] * self.sigma_vec[it]
                self._rhos  += alpha[it] * self.rho_vec[it]

        else:
            # print("Approx case")
            # Standard case
            self._small_data = False
            # pass string identifier if pynndescent also defines distance metric
            if _HAVE_PYNNDESCENT:
                if self._sparse_data and self.metric in pynn_sparse_named_distances:
                    nn_metric = self.metric
                elif not self._sparse_data and self.metric in pynn_named_distances:
                    nn_metric = self.metric
                else:
                    nn_metric = self._input_distance_func
            else:
                nn_metric = self._input_distance_func

            self.graph_vec = [None]*len(jointX)
            self.sigma_vec = [None]*len(jointX)
            self.rho_vec = [None]*len(jointX)

            for idx, key in enumerate(jointX):
                (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                    jointX[key][index],
                    self._n_neighbors,
                    nn_metric,
                    self._metric_kwds,
                    self.angular_rp_forest,
                    random_state,
                    self.low_memory,
                    use_pynndescent=True,
                    verbose=self.verbose,
                )

                self.graph_vec[idx], self.sigma_vec[idx], self.rho_vec[idx] = fuzzy_simplicial_set(
                    jointX[key][index],
                    self.n_neighbors,
                    random_state,
                    nn_metric,
                    self._metric_kwds,
                    self._knn_indices,
                    self._knn_dists,
                    self.angular_rp_forest,
                    self.set_op_mix_ratio,
                    self.local_connectivity,
                    True,
                    self.verbose,
                )
            self.graph_  = alpha[0] * self.graph_vec[0]
            self._sigmas = alpha[0] * self.sigma_vec[0]
            self._rhos   = alpha[0] * self.rho_vec[0]
            # self._sigmas = np.average(np.array(self.sigma_vec))
            # self._rhos = np.average(np.array(self.rho_vec))
            for it in range(1, len(jointX)):
                self.graph_ += alpha[it] * self.graph_vec[it]
                self._sigmas += alpha[it] * self.sigma_vec[it]
                self._rhos  += alpha[it] * self.rho_vec[it]

            if not _HAVE_PYNNDESCENT:
                print ("Not implemented, need to install Pynndescent")
                # self._search_graph = scipy.sparse.lil_matrix(
                #     (X[index].shape[0], X[index].shape[0]), dtype=np.int8
                # )
                # _rows = []
                # _data = []
                # for i in self._knn_indices:
                #     _non_neg = i[i >= 0]
                #     _rows.append(_non_neg.tolist())
                #     _data.append(np.ones(_non_neg.shape[0], dtype=np.int8).tolist())
                #
                # self._search_graph.rows = np.empty(len(_rows), dtype=object)
                # self._search_graph.rows[:] = _rows
                # self._search_graph.data = np.empty(len(_data), dtype=object)
                # self._search_graph.data[:] = _data
                # self._search_graph = self._search_graph.maximum(
                #     self._search_graph.transpose()
                # ).tocsr()
                #
                # if (self.metric != "precomputed") and (len(self._metric_kwds) > 0):
                #     # Create a partial function for distances with arguments
                #     _distance_func = self._input_distance_func
                #     _dist_args = tuple(self._metric_kwds.values())
                #     if self._sparse_data:
                #
                #         @numba.njit()
                #         def _partial_dist_func(ind1, data1, ind2, data2):
                #             return _distance_func(ind1, data1, ind2, data2, *_dist_args)
                #
                #         self._input_distance_func = _partial_dist_func
                #     else:
                #
                #         @numba.njit()
                #         def _partial_dist_func(x, y):
                #             return _distance_func(x, y, *_dist_args)
                #
                #         self._input_distance_func = _partial_dist_func

        # Currently not checking if any duplicate points have differing labels
        # Might be worth throwing a warning...

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct embedding")

        self.embedding_ = simplicial_set_embedding(
            self._raw_data[index],  # JH why raw data?
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
        )[inverse]

        if self.verbose:
            print(ts() + " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

        return self

    def fit_transform(self, X, alpha):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, alpha)
        return self.embedding_

class JUMAP(JUMAPBASE):
    def CE(self, P, Q):
        """
        Compute Cross-Entropy (CE) 
        """
        # return np.sum(P * np.log(P + 0.001) + (1-P) * np.log(1 - P + 0.001) - P * np.log(Q + 0.001) - (1 - P) * np.log(1 - Q + 0.001))
        return  np.sum(-P * np.log(Q + 0.001) - (1 - P) * np.log(1 - Q + 0.001))
    def prob_low_dim(self, Y, a, b):
        return 1.0/(1.0 + a*np.power(pairwise_distances(Y), 2*b))

    def fit_transform(self, X, method = "uniform", ld = 1.0, max_iter = 10):
        alpha = [1/len(X)]*len(X)
        if method == "uniform":
            self.fit(X, alpha)
            return self.embedding_ 
        else:
            for it in range(max_iter):
                # start = time.time()
                self.fit(X, alpha)
                # print("Fit time: ", time.time() - start)
                # start = time.time()
                Q = self.prob_low_dim(self.embedding_, self._a, self._b)
                # print("Q time: ", time.time() - start)
                # n_samples = Q.shape[0]
                # start = time.time()
                obj_vec = np.array([self.CE(self.graph_vec[i].toarray(), Q) for i in range(len(X))])
                # print("Ob time: ", time.time() - start)
                if it == 0:
                    scaleOBJ = 0.1*np.max(obj_vec) # use this to scale the objective function
                obj_vec = obj_vec/(0.001+ scaleOBJ)

                alpha = np.exp(-obj_vec/ld - 1)
                alpha = alpha/np.sum(alpha)

                alpha_np = np.array(alpha)
                self.obj_value = np.inner(obj_vec, alpha_np) + ld * np.inner(alpha_np, np.log(alpha_np))
                # print(self.obj_value)
                self.init = self.embedding_
                # print(alpha)
            self.alpha = alpha
            return self.embedding_ 
        



#sparse#######

# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause


locale.setlocale(locale.LC_NUMERIC, "C")

# Just reproduce a simpler version of numpy unique (not numba supported yet)
@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
    return aux[flag]


# Just reproduce a simpler version of numpy union1d (not numba supported yet)
@numba.njit()
def arr_union(ar1, ar2):
    if ar1.shape[0] == 0:
        return ar2
    elif ar2.shape[0] == 0:
        return ar1
    else:
        return arr_unique(np.concatenate((ar1, ar2)))


# Just reproduce a simpler version of numpy intersect1d (not numba supported
# yet)
@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def sparse_sum(ind1, data1, ind2, data2):
    result_ind = arr_union(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = ind1[i1]
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = ind2[i2]
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit()
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = arr_intersect(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def general_sset_intersection(
    indptr1,
    indices1,
    data1,
    indptr2,
    indices2,
    data2,
    result_row,
    result_col,
    result_val,
    mix_weight=0.5,
):

    left_min = max(data1.min() / 2.0, 1.0e-8)
    right_min = max(data2.min() / 2.0, 1.0e-8)

    for idx in range(result_row.shape[0]):
        i = result_row[idx]
        j = result_col[idx]

        left_val = left_min
        for k in range(indptr1[i], indptr1[i + 1]):
            if indices1[k] == j:
                left_val = data1[k]

        right_val = right_min
        for k in range(indptr2[i], indptr2[i + 1]):
            if indices2[k] == j:
                right_val = data2[k]

        if left_val > left_min or right_val > right_min:
            if mix_weight < 0.5:
                result_val[idx] = left_val * pow(
                    right_val, mix_weight / (1.0 - mix_weight)
                )
            else:
                result_val[idx] = (
                    pow(left_val, (1.0 - mix_weight) / mix_weight) * right_val
                )

    return


@numba.njit()
def sparse_euclidean(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += aux_data[i] ** 2
    return np.sqrt(result)


@numba.njit()
def sparse_manhattan(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i])
    return result


@numba.njit()
def sparse_chebyshev(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result = max(result, np.abs(aux_data[i]))
    return result


@numba.njit()
def sparse_minkowski(ind1, data1, ind2, data2, p=2.0):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i]) ** p
    return result ** (1.0 / p)


@numba.njit()
def sparse_hamming(ind1, data1, ind2, data2, n_features):
    num_not_equal = sparse_diff(ind1, data1, ind2, data2)[0].shape[0]
    return float(num_not_equal) / n_features


@numba.njit()
def sparse_canberra(ind1, data1, ind2, data2):
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)
    denom_data = 1.0 / denom_data
    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    val_inds, val_data = sparse_mul(numer_inds, numer_data, denom_inds, denom_data)

    return np.sum(val_data)


@numba.njit()
def sparse_bray_curtis(ind1, data1, ind2, data2):  # pragma: no cover
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)

    if denom_data.shape[0] == 0:
        return 0.0

    denominator = np.sum(denom_data)

    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    numerator = np.sum(numer_data)

    return float(numerator) / denominator


@numba.njit()
def sparse_jaccard(ind1, data1, ind2, data2):
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_equal = arr_intersect(ind1, ind2).shape[0]

    if num_non_zero == 0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def sparse_matching(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return float(num_not_equal) / n_features


@numba.njit()
def sparse_dice(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def sparse_kulsinski(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + n_features) / (
            num_not_equal + n_features
        )


@numba.njit()
def sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_russellrao(ind1, data1, ind2, data2, n_features):
    if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2):
        return 0.0

    num_true_true = arr_intersect(ind1, ind2).shape[0]

    if num_true_true == np.sum(data1 != 0) and num_true_true == np.sum(data2 != 0):
        return 0.0
    else:
        return float(n_features - num_true_true) / (n_features)


@numba.njit()
def sparse_sokal_michener(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_sokal_sneath(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def sparse_cosine(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = norm(data1)
    norm2 = norm(data2)

    for i in range(aux_data.shape[0]):
        result += aux_data[i]

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    else:
        return 1.0 - (result / (norm1 * norm2))


@numba.njit()
def sparse_hellinger(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = np.sum(data1)
    norm2 = np.sum(data2)
    sqrt_norm_prod = np.sqrt(norm1 * norm2)

    for i in range(aux_data.shape[0]):
        result += np.sqrt(aux_data[i])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    elif result > sqrt_norm_prod:
        return 0.0
    else:
        return np.sqrt(1.0 - (result / sqrt_norm_prod))


@numba.njit()
def sparse_correlation(ind1, data1, ind2, data2, n_features):

    mu_x = 0.0
    mu_y = 0.0
    dot_product = 0.0

    if ind1.shape[0] == 0 and ind2.shape[0] == 0:
        return 0.0
    elif ind1.shape[0] == 0 or ind2.shape[0] == 0:
        return 1.0

    for i in range(data1.shape[0]):
        mu_x += data1[i]
    for i in range(data2.shape[0]):
        mu_y += data2[i]

    mu_x /= n_features
    mu_y /= n_features

    shifted_data1 = np.empty(data1.shape[0], dtype=np.float32)
    shifted_data2 = np.empty(data2.shape[0], dtype=np.float32)

    for i in range(data1.shape[0]):
        shifted_data1[i] = data1[i] - mu_x
    for i in range(data2.shape[0]):
        shifted_data2[i] = data2[i] - mu_y

    norm1 = np.sqrt(
        (norm(shifted_data1) ** 2) + (n_features - ind1.shape[0]) * (mu_x ** 2)
    )
    norm2 = np.sqrt(
        (norm(shifted_data2) ** 2) + (n_features - ind2.shape[0]) * (mu_y ** 2)
    )

    dot_prod_inds, dot_prod_data = sparse_mul(ind1, shifted_data1, ind2, shifted_data2)

    common_indices = set(dot_prod_inds)

    for i in range(dot_prod_data.shape[0]):
        dot_product += dot_prod_data[i]

    for i in range(ind1.shape[0]):
        if ind1[i] not in common_indices:
            dot_product -= shifted_data1[i] * (mu_y)

    for i in range(ind2.shape[0]):
        if ind2[i] not in common_indices:
            dot_product -= shifted_data2[i] * (mu_x)

    all_indices = arr_union(ind1, ind2)
    dot_product += mu_x * mu_y * (n_features - all_indices.shape[0])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / (norm1 * norm2))


@numba.njit()
def approx_log_Gamma(x):
    if x == 1:
        return 0
    #    x2= 1/(x*x);
    return (
        x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi / x) + 1.0 / (x * 12.0)
    )  # + x2*(-1.0/360.0 + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)


#                + x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 + x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0)
#                + x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))


@numba.njit()
def log_beta(x, y):
    a = min(x, y)
    b = max(x, y)
    if b < 5:
        value = -np.log(b)
        for i in range(1, int(a)):
            value += np.log(i) - np.log(b + i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


@numba.njit()
def log_single_beta(x):
    return (
        np.log(2.0) * (-2.0 * x + 0.5) + 0.5 * np.log(2.0 * np.pi / x) + 0.125 / x
    )  # + x2*(-1.0/192.0 + x2* (1.0/640.0 + x2*(-17.0/(14336.0)


#                + x2*(31.0/18432.0 + x2*(-691.0/180224.0 + x2*(5461.0/425984.0 + x2*(-929569.0/15728640.0 + x2*(3189151.0/8912896.0 + x2*(-221930581.0/79691776.0)
#                + x2*(4722116521.0/176160768.0 + x2*(-968383680827.0/3087007744.0 + x2*(14717667114151.0/3355443200.0 ))))))))))))


@numba.njit()
def sparse_ll_dirichlet(ind1, data1, ind2, data2):
    # The probability of rolling data2 in sum(data2) trials on a die that rolled data1 in sum(data1) trials
    n1 = np.sum(data1)
    n2 = np.sum(data2)

    if n1 == 0 and n2 == 0:
        return 0.0
    elif n1 == 0 or n2 == 0:
        return 1e8

    log_b = 0.0
    i1 = 0
    i2 = 0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            if data1[i1] * data2[i2] != 0:
                log_b += log_beta(data1[i1], data2[i2])
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    self_denom1 = 0.0
    for d1 in data1:

        self_denom1 += log_single_beta(d1)

    self_denom2 = 0.0
    for d2 in data2:
        self_denom2 += log_single_beta(d2)

    return np.sqrt(
        1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
    )


sparse_named_distances = {
    # general minkowski distances
    "euclidean": sparse_euclidean,
    "manhattan": sparse_manhattan,
    "l1": sparse_manhattan,
    "taxicab": sparse_manhattan,
    "chebyshev": sparse_chebyshev,
    "linf": sparse_chebyshev,
    "linfty": sparse_chebyshev,
    "linfinity": sparse_chebyshev,
    "minkowski": sparse_minkowski,
    # Other distances
    "canberra": sparse_canberra,
    "ll_dirichlet": sparse_ll_dirichlet,
    # 'braycurtis': sparse_bray_curtis,
    # Binary distances
    "hamming": sparse_hamming,
    "jaccard": sparse_jaccard,
    "dice": sparse_dice,
    "matching": sparse_matching,
    "kulsinski": sparse_kulsinski,
    "rogerstanimoto": sparse_rogers_tanimoto,
    "russellrao": sparse_russellrao,
    "sokalmichener": sparse_sokal_michener,
    "sokalsneath": sparse_sokal_sneath,
    "cosine": sparse_cosine,
    "correlation": sparse_correlation,
    "hellinger": sparse_hellinger,
}

sparse_need_n_features = (
    "hamming",
    "matching",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "correlation",
)

SPARSE_SPECIAL_METRICS = {
    sparse_hellinger: "hellinger",
    sparse_ll_dirichlet: "ll_dirichlet",
}






#sparse_nndescent################


locale.setlocale(locale.LC_NUMERIC, "C")


@numba.njit(fastmath=True)
def sparse_init_rp_tree(
    inds, indptr, data, sparse_dist, current_graph, leaf_array, tried=None
):
    if tried is None:
        tried = set([(-1, -1)])

    for n in range(leaf_array.shape[0]):
        for i in range(leaf_array.shape[1]):
            p = leaf_array[n, i]
            if p < 0:
                break
            for j in range(i + 1, leaf_array.shape[1]):
                q = leaf_array[n, j]
                if q < 0:
                    break
                if (p, q) in tried:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]
                d = sparse_dist(from_inds, from_data, to_inds, to_data)
                heap_push(current_graph, p, d, q, 1)
                tried.add((p, q))
                if p != q:
                    heap_push(current_graph, q, d, p, 1)
                    tried.add((q, p))


@numba.njit(fastmath=True)
def sparse_nn_descent_internal_low_memory(
    current_graph,
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    max_candidates=50,
    sparse_dist=sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit(fastmath=True)
def sparse_nn_descent_internal_high_memory(
    current_graph,
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    tried,
    max_candidates=50,
    sparse_dist=sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit(fastmath=True)
def sparse_nn_descent(
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    max_candidates=50,
    sparse_dist=sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    low_memory=False,
    rp_tree_init=True,
    leaf_array=None,
    verbose=False,
):

    tried = set([(-1, -1)])

    current_graph = make_heap(n_vertices, n_neighbors)
    for i in range(n_vertices):
        indices = rejection_sample(n_neighbors, n_vertices, rng_state)
        for j in range(indices.shape[0]):

            from_inds = inds[indptr[i] : indptr[i + 1]]
            from_data = data[indptr[i] : indptr[i + 1]]

            to_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            to_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)

            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        sparse_init_rp_tree(
            inds, indptr, data, sparse_dist, current_graph, leaf_array, tried=tried,
        )

    if low_memory:
        sparse_nn_descent_internal_low_memory(
            current_graph,
            inds,
            indptr,
            data,
            n_vertices,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            sparse_dist=sparse_dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )
    else:
        sparse_nn_descent_internal_high_memory(
            current_graph,
            inds,
            indptr,
            data,
            n_vertices,
            n_neighbors,
            rng_state,
            tried,
            max_candidates=max_candidates,
            sparse_dist=sparse_dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )

    return deheap_sort(current_graph)


@numba.njit()
def sparse_init_from_random(
    n_neighbors,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    heap,
    rng_state,
    sparse_dist,
):
    for i in range(query_indptr.shape[0] - 1):
        indices = rejection_sample(n_neighbors, indptr.shape[0] - 1, rng_state)

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue

            from_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            from_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)
            heap_push(heap, i, d, indices[j], 1)
    return


@numba.njit()
def sparse_init_from_tree(
    tree,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    heap,
    rng_state,
    sparse_dist,
):
    for i in range(query_indptr.shape[0] - 1):

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        indices = search_sparse_flat_tree(
            to_inds,
            to_data,
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            from_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            from_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)
            heap_push(heap, i, d, indices[j], 1)

    return


def sparse_initialise_search(
    forest,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    n_neighbors,
    rng_state,
    sparse_dist,
):
    results = make_heap(query_indptr.shape[0] - 1, n_neighbors)
    sparse_init_from_random(
        n_neighbors,
        inds,
        indptr,
        data,
        query_inds,
        query_indptr,
        query_data,
        results,
        rng_state,
        sparse_dist,
    )
    if forest is not None:
        for tree in forest:
            sparse_init_from_tree(
                tree,
                inds,
                indptr,
                data,
                query_inds,
                query_indptr,
                query_data,
                results,
                rng_state,
                sparse_dist,
            )

    return results


@numba.njit(parallel=True)
def sparse_initialized_nnd_search(
    inds,
    indptr,
    data,
    search_indptr,
    search_inds,
    initialization,
    query_inds,
    query_indptr,
    query_data,
    sparse_dist,
):
    for i in numba.prange(query_indptr.shape[0] - 1):

        tried = set(initialization[0, i])

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        while True:

            # Find smallest flagged vertex
            vertex = smallest_flagged(initialization, i)

            if vertex == -1:
                break
            candidates = search_inds[search_indptr[vertex] : search_indptr[vertex + 1]]

            for j in range(candidates.shape[0]):
                if (
                    candidates[j] == vertex
                    or candidates[j] == -1
                    or candidates[j] in tried
                ):
                    continue

                from_inds = inds[indptr[candidates[j]] : indptr[candidates[j] + 1]]
                from_data = data[indptr[candidates[j]] : indptr[candidates[j] + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data)
                unchecked_heap_push(initialization, i, d, candidates[j], 1)
                tried.add(candidates[j])

    return initialization




#utils################

@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.
    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.
    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])")
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]
    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit()
def norm(vec):
    """Compute the (standard l2) norm of a vector.
    Parameters
    ----------
    vec: array of shape (dim,)
    Returns
    -------
    The l2 norm of vec.
    """
    result = 0.0
    for i in range(vec.shape[0]):
        result += vec[i] ** 2
    return np.sqrt(result)


@numba.njit()
def rejection_sample(n_samples, pool_size, rng_state):
    """Generate n_samples many integers from 0 to pool_size such that no
    integer is selected twice. The duplication constraint is achieved via
    rejection sampling.
    Parameters
    ----------
    n_samples: int
        The number of random samples to select from the pool
    pool_size: int
        The size of the total pool of candidates to sample from
    rng_state: array of int64, shape (3,)
        Internal state of the random number generator
    Returns
    -------
    sample: array of shape(n_samples,)
        The ``n_samples`` randomly selected elements from the pool.
    """
    result = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        reject_sample = True
        j = 0
        while reject_sample:
            j = tau_rand_int(rng_state) % pool_size
            for k in range(i):
                if j == result[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit()
def make_heap(n_points, size):
    """Constructor for the numba enabled heap objects. The heaps are used
    for approximate nearest neighbor search, maintaining a list of potential
    neighbors sorted by their distance. We also flag if potential neighbors
    are newly added to the list or not. Internally this is stored as
    a single ndarray; the first axis determines whether we are looking at the
    array of candidate indices, the array of distances, or the flag array for
    whether elements are new or not. Each of these arrays are of shape
    (``n_points``, ``size``)
    Parameters
    ----------
    n_points: int
        The number of data points to track in the heap.
    size: int
        The number of items to keep on the heap for each data point.
    Returns
    -------
    heap: An ndarray suitable for passing to other numba enabled heap functions.
    """
    result = np.zeros(
        (np.int64(3), np.int64(n_points), np.int64(size)), dtype=np.float64
    )
    result[0] = -1
    result[1] = np.infty
    result[2] = 0

    return result


@numba.njit("i8(f8[:,:,:],i8,f8,i8,i8)")
def heap_push(heap, row, weight, index, flag):
    """Push a new element onto the heap. The heap stores potential neighbors
    for each data point. The ``row`` parameter determines which data point we
    are addressing, the ``weight`` determines the distance (for heap sorting),
    the ``index`` is the element to add, and the flag determines whether this
    is to be considered a new addition.
    Parameters
    ----------
    heap: ndarray generated by ``make_heap``
        The heap object to push into
    row: int
        Which actual heap within the heap object to push to
    weight: float
        The priority value of the element to push onto the heap
    index: int
        The actual value to be pushed
    flag: int
        Whether to flag the newly added element or not.
    Returns
    -------
    success: The number of new elements successfully pushed into the heap.
    """
    row = int(row)
    indices = heap[0, row]
    weights = heap[1, row]
    is_new = heap[2, row]

    if weight >= weights[0]:
        return 0

    # break if we already have this element.
    for i in range(indices.shape[0]):
        if index == indices[i]:
            return 0

    # insert val at position zero
    weights[0] = weight
    indices[0] = index
    is_new[0] = flag

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap.shape[2]:
            break
        elif ic2 >= heap.shape[2]:
            if weights[ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[ic1] >= weights[ic2]:
            if weight < weights[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[ic2]:
                i_swap = ic2
            else:
                break

        weights[i] = weights[i_swap]
        indices[i] = indices[i_swap]
        is_new[i] = is_new[i_swap]

        i = i_swap

    weights[i] = weight
    indices[i] = index
    is_new[i] = flag

    return 1


@numba.njit("i8(f8[:,:,:],i8,f8,i8,i8)")
def unchecked_heap_push(heap, row, weight, index, flag):
    """Push a new element onto the heap. The heap stores potential neighbors
    for each data point. The ``row`` parameter determines which data point we
    are addressing, the ``weight`` determines the distance (for heap sorting),
    the ``index`` is the element to add, and the flag determines whether this
    is to be considered a new addition.
    Parameters
    ----------
    heap: ndarray generated by ``make_heap``
        The heap object to push into
    row: int
        Which actual heap within the heap object to push to
    weight: float
        The priority value of the element to push onto the heap
    index: int
        The actual value to be pushed
    flag: int
        Whether to flag the newly added element or not.
    Returns
    -------
    success: The number of new elements successfully pushed into the heap.
    """
    if weight >= heap[1, row, 0]:
        return 0

    indices = heap[0, row]
    weights = heap[1, row]
    is_new = heap[2, row]

    # insert val at position zero
    weights[0] = weight
    indices[0] = index
    is_new[0] = flag

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap.shape[2]:
            break
        elif ic2 >= heap.shape[2]:
            if weights[ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[ic1] >= weights[ic2]:
            if weight < weights[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[ic2]:
                i_swap = ic2
            else:
                break

        weights[i] = weights[i_swap]
        indices[i] = indices[i_swap]
        is_new[i] = is_new[i_swap]

        i = i_swap

    weights[i] = weight
    indices[i] = index
    is_new[i] = flag

    return 1


@numba.njit()
def siftdown(heap1, heap2, elt):
    """Restore the heap property for a heap with an out of place element
    at position ``elt``. This works with a heap pair where heap1 carries
    the weights and heap2 holds the corresponding elements."""
    while elt * 2 + 1 < heap1.shape[0]:
        left_child = elt * 2 + 1
        right_child = left_child + 1
        swap = elt

        if heap1[swap] < heap1[left_child]:
            swap = left_child

        if right_child < heap1.shape[0] and heap1[swap] < heap1[right_child]:
            swap = right_child

        if swap == elt:
            break
        else:
            heap1[elt], heap1[swap] = (heap1[swap], heap1[elt])
            heap2[elt], heap2[swap] = (heap2[swap], heap2[elt])
            elt = swap


@numba.njit()
def deheap_sort(heap):
    """Given an array of heaps (of indices and weights), unpack the heap
    out to give and array of sorted lists of indices and weights by increasing
    weight. This is effectively just the second half of heap sort (the first
    half not being required since we already have the data in a heap).
    Parameters
    ----------
    heap : array of shape (3, n_samples, n_neighbors)
        The heap to turn into sorted lists.
    Returns
    -------
    indices, weights: arrays of shape (n_samples, n_neighbors)
        The indices and weights sorted by increasing weight.
    """
    indices = heap[0]
    weights = heap[1]

    for i in range(indices.shape[0]):

        ind_heap = indices[i]
        dist_heap = weights[i]

        for j in range(ind_heap.shape[0] - 1):
            ind_heap[0], ind_heap[ind_heap.shape[0] - j - 1] = (
                ind_heap[ind_heap.shape[0] - j - 1],
                ind_heap[0],
            )
            dist_heap[0], dist_heap[dist_heap.shape[0] - j - 1] = (
                dist_heap[dist_heap.shape[0] - j - 1],
                dist_heap[0],
            )

            siftdown(
                dist_heap[: dist_heap.shape[0] - j - 1],
                ind_heap[: ind_heap.shape[0] - j - 1],
                0,
            )

    return indices.astype(np.int64), weights


@numba.njit("i8(f8[:, :, :],i8)")
def smallest_flagged(heap, row):
    """Search the heap for the smallest element that is
    still flagged.
    Parameters
    ----------
    heap: array of shape (3, n_samples, n_neighbors)
        The heaps to search
    row: int
        Which of the heaps to search
    Returns
    -------
    index: int
        The index of the smallest flagged element
        of the ``row``th heap, or -1 if no flagged
        elements remain in the heap.
    """
    ind = heap[0, row]
    dist = heap[1, row]
    flag = heap[2, row]

    min_dist = np.inf
    result_index = -1

    for i in range(ind.shape[0]):
        if flag[i] == 1 and dist[i] < min_dist:
            min_dist = dist[i]
            result_index = i

    if result_index >= 0:
        flag[result_index] = 0.0
        return int(ind[result_index])
    else:
        return -1


@numba.njit(parallel=True)
def build_candidates(current_graph, n_vertices, n_neighbors, max_candidates, rng_state):
    """Build a heap of candidate neighbors for nearest neighbor descent. For
    each vertex the candidate neighbors are any current neighbors, and any
    vertices that have the vertex as one of their nearest neighbors.
    Parameters
    ----------
    current_graph: heap
        The current state of the graph for nearest neighbor descent.
    n_vertices: int
        The total number of vertices in the graph.
    n_neighbors: int
        The number of neighbor edges per node in the current graph.
    max_candidates: int
        The maximum number of new candidate neighbors.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    candidate_neighbors: A heap with an array of (randomly sorted) candidate
    neighbors for each vertex in the graph.
    """
    candidate_neighbors = make_heap(n_vertices, max_candidates)
    for i in range(n_vertices):
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = tau_rand(rng_state)
            heap_push(candidate_neighbors, i, d, idx, isn)
            heap_push(candidate_neighbors, idx, d, i, isn)
            current_graph[2, i, j] = 0

    return candidate_neighbors


@numba.njit()
def new_build_candidates(
    current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho=0.5
):  # pragma: no cover
    """Build a heap of candidate neighbors for nearest neighbor descent. For
    each vertex the candidate neighbors are any current neighbors, and any
    vertices that have the vertex as one of their nearest neighbors.
    Parameters
    ----------
    current_graph: heap
        The current state of the graph for nearest neighbor descent.
    n_vertices: int
        The total number of vertices in the graph.
    n_neighbors: int
        The number of neighbor edges per node in the current graph.
    max_candidates: int
        The maximum number of new candidate neighbors.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    candidate_neighbors: A heap with an array of (randomly sorted) candidate
    neighbors for each vertex in the graph.
    """
    new_candidate_neighbors = make_heap(n_vertices, max_candidates)
    old_candidate_neighbors = make_heap(n_vertices, max_candidates)

    for i in range(n_vertices):
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = tau_rand(rng_state)
            if tau_rand(rng_state) < rho:
                c = 0
                if isn:
                    c += heap_push(new_candidate_neighbors, i, d, idx, isn)
                    c += heap_push(new_candidate_neighbors, idx, d, i, isn)
                else:
                    heap_push(old_candidate_neighbors, i, d, idx, isn)
                    heap_push(old_candidate_neighbors, idx, d, i, isn)

                if c > 0:
                    current_graph[2, i, j] = 0

    return new_candidate_neighbors, old_candidate_neighbors


@numba.njit(parallel=True)
def submatrix(dmat, indices_col, n_neighbors):
    """Return a submatrix given an orginal matrix and the indices to keep.
    Parameters
    ----------
    dmat: array, shape (n_samples, n_samples)
        Original matrix.
    indices_col: array, shape (n_samples, n_neighbors)
        Indices to keep. Each row consists of the indices of the columns.
    n_neighbors: int
        Number of neighbors.
    Returns
    -------
    submat: array, shape (n_samples, n_neighbors)
        The corresponding submatrix.
    """
    n_samples_transform, n_samples_fit = dmat.shape
    submat = np.zeros((n_samples_transform, n_neighbors), dtype=dmat.dtype)
    for i in numba.prange(n_samples_transform):
        for j in numba.prange(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())


# I'm not enough of a numba ninja to numba this successfully.
# np.arrays of lists, which are objects...
def csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True):
    """Find the unique elements of a sparse csr matrix.
    We don't explicitly construct the unique matrix leaving that to the user
    who may not want to duplicate a massive array in memory.
    Returns the indices of the input array that give the unique values.
    Returns the indices of the unique array that reconstructs the input array.
    Returns the number of times each unique row appears in the input matrix.
    matrix: a csr matrix
    return_index = bool, optional
        If true, return the row indices of 'matrix'
    return_inverse: bool, optional
        If true, return the the indices of the unique array that can be
           used to reconstruct 'matrix'.
    return_counts = bool, optional
        If true, returns the number of times each unique item appears in 'matrix'
    The unique matrix can computed via
    unique_matrix = matrix[index]
    and the original matrix reconstructed via
    unique_matrix[inverse]
    """
    lil_matrix = matrix.tolil()
    rows = [x + y for x, y in zip(lil_matrix.rows, lil_matrix.data)]
    return_values = return_counts + return_inverse + return_index
    return np.unique(
        rows,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )[1 : (return_values + 1)]








#rp_tree##################


locale.setlocale(locale.LC_NUMERIC, "C")

# Used for a floating point "nearly zero" comparison
EPS = 1e-8

RandomProjectionTreeNode = namedtuple(
    "RandomProjectionTreeNode",
    ["indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
)

FlatTree = namedtuple("FlatTree", ["hyperplanes", "offsets", "children", "indices"])


@numba.njit(fastmath=True)
def angular_random_projection_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split
    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = norm(data[left])
    right_norm = norm(data[right])

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, None


@numba.njit(fastmath=True, nogil=True)
def euclidean_random_projection_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses euclidean distance to determine the hyperplane
    and which side each data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split
    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, hyperplane_offset


@numba.njit(fastmath=True)
def sparse_angular_random_projection_split(inds, indptr, data, indices, rng_state):
    """Given a set of ``indices`` for data points from a sparse data set
    presented in csr sparse format as inds, indptr and data, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.
    Parameters
    ----------
    inds: array
        CSR format index array of the matrix
    indptr: array
        CSR format index pointer array of the matrix
    data: array
        CSR format data array of the matrix
    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    normalized_left_data = left_data / left_norm
    normalized_right_data = right_data / right_norm
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, normalized_left_data, right_inds, normalized_right_data
    )

    hyperplane_norm = norm(hyperplane_data)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0
    for d in range(hyperplane_data.shape[0]):
        hyperplane_data[d] = hyperplane_data[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0

        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, None


@numba.njit(fastmath=True)
def sparse_euclidean_random_projection_split(inds, indptr, data, indices, rng_state):
    """Given a set of ``indices`` for data points from a sparse data set
    presented in csr sparse format as inds, indptr and data, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.
    Parameters
    ----------
    inds: array
        CSR format index array of the matrix
    indptr: array
        CSR format index pointer array of the matrix
    data: array
        CSR format data array of the matrix
    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds, hyperplane_data, offset_inds, offset_data
    )

    for d in range(offset_data.shape[0]):
        hyperplane_offset -= offset_data[d]

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, hyperplane_offset


def make_euclidean_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = euclidean_random_projection_split(data, indices, rng_state)

        left_node = make_euclidean_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_euclidean_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_angular_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = angular_random_projection_split(data, indices, rng_state)

        left_node = make_angular_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_angular_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_euclidean_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_euclidean_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_euclidean_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_euclidean_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_angular_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_angular_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_angular_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_angular_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_tree(data, rng_state, leaf_size=30, angular=False):
    """Construct a random projection tree based on ``data`` with leaves
    of size at most ``leaf_size``.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    leaf_size: int (optional, default 30)
        The maximum size of any leaf node in the tree. Any node in the tree
        with more than ``leaf_size`` will be split further to create child
        nodes.
    angular: bool (optional, default False)
        Whether to use cosine/angular distance to create splits in the tree,
        or euclidean distance.
    Returns
    -------
    node: RandomProjectionTreeNode
        A random projection tree node which links to its child nodes. This
        provides the full tree below the returned node.
    """
    is_sparse = scipy.sparse.isspmatrix_csr(data)
    indices = np.arange(data.shape[0])

    # Make a tree recursively until we get below the leaf size
    if is_sparse:
        inds = data.indices
        indptr = data.indptr
        spdata = data.data

        if angular:
            return make_sparse_angular_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
        else:
            return make_sparse_euclidean_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
    else:
        if angular:
            return make_angular_tree(data, indices, rng_state, leaf_size)
        else:
            return make_euclidean_tree(data, indices, rng_state, leaf_size)


def num_nodes(tree):
    """Determine the number of nodes in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return 1 + num_nodes(tree.left_child) + num_nodes(tree.right_child)


def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)


def max_sparse_hyperplane_size(tree):
    """Determine the most number on non zeros in a hyperplane entry"""
    if tree.is_leaf:
        return 0
    else:
        return max(
            tree.hyperplane.shape[1],
            max_sparse_hyperplane_size(tree.left_child),
            max_sparse_hyperplane_size(tree.right_child),
        )


def recursive_flatten(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_num
):
    if tree.is_leaf:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : tree.indices.shape[0]] = tree.indices
        leaf_num += 1
        return node_num, leaf_num
    else:
        if len(tree.hyperplane.shape) > 1:
            # sparse case
            hyperplanes[node_num][:, : tree.hyperplane.shape[1]] = tree.hyperplane
        else:
            hyperplanes[node_num] = tree.hyperplane
        offsets[node_num] = tree.offset
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_flatten(
            tree.left_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_flatten(
            tree.right_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        return node_num, leaf_num


def flatten_tree(tree, leaf_size):
    n_nodes = num_nodes(tree)
    n_leaves = num_leaves(tree)

    if len(tree.hyperplane.shape) > 1:
        # sparse case
        max_hyperplane_nnz = max_sparse_hyperplane_size(tree)
        hyperplanes = np.zeros(
            (n_nodes, tree.hyperplane.shape[0], max_hyperplane_nnz), dtype=np.float32
        )
    else:
        hyperplanes = np.zeros((n_nodes, tree.hyperplane.shape[0]), dtype=np.float32)

    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
    indices = -1 * np.ones((n_leaves, leaf_size), dtype=np.int64)
    recursive_flatten(tree, hyperplanes, offsets, children, indices, 0, 0)
    return FlatTree(hyperplanes, offsets, children, indices)


@numba.njit()
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    for d in range(point.shape[0]):
        margin += hyperplane[d] * point[d]

    if abs(margin) < EPS:
        side = abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit()
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]


@numba.njit()
def sparse_select_side(hyperplane, offset, point_inds, point_data, rng_state):
    margin = offset

    hyperplane_inds = arr_unique(hyperplane[0])
    hyperplane_data = hyperplane[1, : hyperplane_inds.shape[0]]

    aux_inds, aux_data = sparse_mul(
        hyperplane_inds, hyperplane_data, point_inds, point_data
    )

    for d in range(aux_data.shape[0]):
        margin += aux_data[d]

    if margin == 0:
        side = abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit()
def search_sparse_flat_tree(
    point_inds, point_data, hyperplanes, offsets, children, indices, rng_state
):
    node = 0
    while children[node, 0] > 0:
        side = sparse_select_side(
            hyperplanes[node], offsets[node], point_inds, point_data, rng_state
        )
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]


def make_forest(data, n_neighbors, n_trees, rng_state, angular=False):
    """Build a random projection forest with ``n_trees``.
    Parameters
    ----------
    data
    n_neighbors
    n_trees
    rng_state
    angular
    Returns
    -------
    forest: list
        A list of random projection trees.
    """
    result = []
    leaf_size = max(10, n_neighbors)
    try:
        result = [
            flatten_tree(make_tree(data, rng_state, leaf_size, angular), leaf_size)
            for i in range(n_trees)
        ]
    except (RuntimeError, RecursionError, SystemError):
        warn(
            "Random Projection forest initialisation failed due to recursion"
            "limit being reached. Something is a little strange with your "
            "data, and this may take longer than normal to compute."
        )

    return result


def rptree_leaf_array(rp_forest):
    """Generate an array of sets of candidate nearest neighbors by
    constructing a random projection forest and taking the leaves of all the
    trees. Any given tree has leaves that are a set of potential nearest
    neighbors. Given enough trees the set of all such leaves gives a good
    likelihood of getting a good set of nearest neighbors in composite. Since
    such a random projection forest is inexpensive to compute, this can be a
    useful means of seeding other nearest neighbor algorithms.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The data for which to generate nearest neighbor approximations.
    n_neighbors: int
        The number of nearest neighbors to attempt to approximate.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    n_trees: int (optional, default 10)
        The number of trees to build in the forest construction.
    angular: bool (optional, default False)
        Whether to use angular/cosine distance for random projection tree
        construction.
    Returns
    -------
    leaf_array: array of shape (n_leaves, max(10, n_neighbors))
        Each row of leaf array is a list of indices found in a given leaf.
        Since not all leaves are the same size the arrays are padded out with -1
        to ensure we can return a single ndarray.
    """
    if len(rp_forest) > 0:
        leaf_array = np.vstack([tree.indices for tree in rp_forest])
    else:
        leaf_array = np.array([[-1]])

    return leaf_array



#nndescent###########################

@numba.njit(fastmath=True)
def init_current_graph(data, dist, n_neighbors, rng_state):
    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
    return current_graph


@numba.njit(fastmath=True)
def init_rp_tree(data, dist, current_graph, leaf_array, tried=None):
    if tried is None:
        tried = set([(-1, -1)])

    for n in range(leaf_array.shape[0]):
        for i in range(leaf_array.shape[1]):
            p = leaf_array[n, i]
            if p < 0:
                break
            for j in range(i + 1, leaf_array.shape[1]):
                q = leaf_array[n, j]
                if q < 0:
                    break
                if (p, q) in tried:
                    continue
                d = dist(data[p], data[q])
                heap_push(current_graph, p, d, q, 1)
                tried.add((p, q))
                if p != q:
                    heap_push(current_graph, q, d, p, 1)
                    tried.add((q, p))


@numba.njit(fastmath=True)
def nn_descent_internal_low_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    n_vertices = data.shape[0]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    d = dist(data[p], data[q])
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    d = dist(data[p], data[q])
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent_internal_high_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    tried,
    max_candidates=50,
    dist=euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    n_vertices = data.shape[0]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q])
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q])
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
):
    tried = set([(-1, -1)])

    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        init_rp_tree(data, dist, current_graph, leaf_array, tried=tried)

    if low_memory:
        nn_descent_internal_low_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )
    else:
        nn_descent_internal_high_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            tried,
            max_candidates=max_candidates,
            dist=dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )

    return deheap_sort(current_graph)


@numba.njit()
def init_from_random(n_neighbors, data, query_points, heap, rng_state, dist):
    for i in range(query_points.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            d = dist(data[indices[j]], query_points[i])
            heap_push(heap, i, d, indices[j], 1)
    return


@numba.njit()
def init_from_tree(tree, data, query_points, heap, rng_state, dist):
    for i in range(query_points.shape[0]):
        indices = search_flat_tree(
            query_points[i],
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            d = dist(data[indices[j]], query_points[i])
            heap_push(heap, i, d, indices[j], 1)

    return


def initialise_search(forest, data, query_points, n_neighbors, rng_state, dist):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state, dist)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state, dist)

    return results


@numba.njit(parallel=True)
def initialized_nnd_search(data, indptr, indices, initialization, query_points, dist):
    for i in numba.prange(query_points.shape[0]):

        tried = set(initialization[0, i])

        while True:

            # Find smallest flagged vertex
            vertex = smallest_flagged(initialization, i)

            if vertex == -1:
                break
            candidates = indices[indptr[vertex] : indptr[vertex + 1]]
            for j in range(candidates.shape[0]):
                if (
                    candidates[j] == vertex
                    or candidates[j] == -1
                    or candidates[j] in tried
                ):
                    continue
                d = dist(data[candidates[j]], query_points[i])
                unchecked_heap_push(initialization, i, d, candidates[j], 1)
                tried.add(candidates[j])

    return initialization

#spectral##################################


def component_layout(
    data,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'
    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    if metric == "precomputed":
        # cannot compute centroids from precomputed distances
        # instead, compute centroid distances using linkage
        distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
        linkage = metric_kwds.get("linkage", "average")
        if linkage == "average":
            linkage = np.mean
        elif linkage == "complete":
            linkage = np.max
        elif linkage == "single":
            linkage = np.min
        else:
            raise ValueError(
                "Unrecognized linkage '%s'. Please choose from "
                "'average', 'complete', or 'single'" % linkage
            )
        for c_i in range(n_components):
            dm_i = data[component_labels == c_i]
            for c_j in range(c_i + 1, n_components):
                dist = linkage(dm_i[:, component_labels == c_j])
                distance_matrix[c_i, c_j] = dist
                distance_matrix[c_j, c_i] = dist
    else:
        for label in range(n_components):
            component_centroids[label] = data[component_labels == label].mean(axis=0)

        if scipy.sparse.isspmatrix(component_centroids):
            warn(
                "Forcing component centroids to dense; if you are running out of "
                "memory then consider increasing n_neighbors."
            )
            component_centroids = component_centroids.toarray()

        if metric in SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids, metric=metric
            )
        elif metric in SPARSE_SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids, metric=SPARSE_SPECIAL_METRICS[metric]
            )
        else:
            if callable(
                metric
            ) and scipy.sparse.isspmatrix(data):
                function_to_name_mapping = {
                    v: k for k, v in sparse_named_distances.items()
                }
                try:
                    metric_name = function_to_name_mapping[metric]
                except KeyError:
                    raise NotImplementedError(
                        "Multicomponent layout for custom "
                        "sparse metrics is not implemented at "
                        "this time."
                    )
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric_name, **metric_kwds
                )
            else:
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric, **metric_kwds
                )

    affinity_matrix = np.exp(-(distance_matrix ** 2))

    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first fid relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    graph: sparse matrix
        The adjacency matrix of the graph to be emebdded.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        # standard Laplacian
        # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
        # L = D - graph
        # Normalized Laplacian
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / np.sqrt(diag_data),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
        except scipy.sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )

    return result


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim: int
        The dimension of the space into which to embed.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
    
    
    
#layouts##########################

@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)
    Parameters
    ----------
    val: float
        The value to be clamped.
    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    #cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.int64,
    },
)


def rdist(x, y):
    """Reduced Euclidean distance.
    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)
    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel
    )
    for n in range(n_epochs):
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


@numba.njit(fastmath=True)
def optimize_layout_generic(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=euclidean,
    output_metric_kwds=(),
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    weight: array of shape (n_1_simplices)
        The membership weights of the 1-simplices.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )
                _, rev_grad_dist_output = output_metric(
                    other, current, *output_metric_kwds
                )

                if dist_output > 0.0:
                    w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                else:
                    w_l = 1.0
                grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])

                    current[d] += grad_d * alpha
                    if move_other:
                        grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                        other[d] += grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )

                    if dist_output > 0.0:
                        w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                    elif j == k:
                        continue
                    else:
                        w_l = 1.0

                    grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                    for d in range(dim):
                        grad_d = clip(grad_coeff * grad_dist_output[d])
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


@numba.njit(fastmath=True)
def optimize_layout_inverse(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    sigmas,
    rhos,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=euclidean,
    output_metric_kwds=(),
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    weight: array of shape (n_1_simplices)
        The membership weights of the 1-simplices.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )

                w_l = weight[i]
                grad_coeff = -(1 / (w_l * sigmas[k] + 1e-6))

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])

                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )

                    # w_l = 0.0 # for negative samples, the edge does not exist
                    w_h = np.exp(-max(dist_output - rhos[k], 1e-6) / (sigmas[k] + 1e-6))
                    grad_coeff = -gamma * ((0 - w_h) / ((1 - w_h) * sigmas[k] + 1e-6))

                    for d in range(dim):
                        grad_d = clip(grad_coeff * grad_dist_output[d])
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding




#_t_sne##############################


MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.
    Parameters
    ----------
    distances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Parameters
    ----------
    distances : CSR sparse matrix, shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix((conditional_P.ravel(), distances.indices,
                    distances.indptr),
                   shape=(n_samples, n_samples))
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))
    return P


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True, num_threads=1):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : csr sparse matrix, shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    angle : float (default: 0.5)
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    verbose : int
        Verbosity level.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    num_threads : int (optional, default:1)
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    """Batch gradient descent with momentum and individual gains.
    Parameters
    ----------
    objective : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.
    p0 : array-like, shape (n_params,)
        Initial parameter vector.
    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.
    momentum : float, within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.
    min_gain : float, default=0.01
        Minimum individual gain for each parameter.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    verbose : int, default=0
        Verbosity level.
    args : sequence, default=None
        Arguments to pass to objective function.
    kwargs : dict, default=None
        Keyword arguments to pass to objective function.
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i


@_deprecate_positional_args
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric='euclidean'):
    r"""Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.
    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    n_neighbors : int, default=5
        Number of neighbors k that will be considered.
    metric : string, or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, see the
        documentation of argument metric in hoan.pairwise.pairwise_distances
        for a list of available metrics.
        .. versionadded:: 0.20
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    dist_X = pairwise_distances(X, metric=metric)
    if metric == 'precomputed':
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = NearestNeighbors(n_neighbors=n_neighbors).fit(
            X_embedded).kneighbors(return_distance=False)

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    n_samples = X.shape[0]
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis],
                   ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, np.newaxis],
                           ind_X_embedded] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t

class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.
    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].
    Read more in the :ref:`User Guide <t_sne>`.
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    metric : string or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    init : string or numpy array, default="random"
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term: `Glossary <random_state>`.
    method : string, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.
        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.
    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.22
    square_distances : {True, 'legacy'}, default='legacy'
        Whether TSNE should square the distance values. ``'legacy'`` means
        that distance values are squared only when ``metric="euclidean"``.
        ``True`` means that distance values are squared for all metrics.
        .. versionadded:: 0.24
           Added to provide backward compatibility during deprecation of
           legacy squaring behavior.
        .. deprecated:: 0.24
           Legacy squaring behavior was deprecated in 0.24. The ``'legacy'``
           value will be removed in 0.26, at which point the default value will
           change to ``True``.
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from hoan.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    References
    ----------
    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/
    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
    """
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 n_jobs=None, square_distances='legacy'):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        # TODO Revisit deprecation of square_distances for 0.26-0.28 (#12401)
        self.square_distances = square_distances

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.square_distances not in [True, 'legacy']:
            raise ValueError("'square_distances' must be True or 'legacy'.")
        if self.metric != "euclidean" and self.square_distances is not True:
            warnings.warn(("'square_distances' has been introduced in 0.24"
                           "to help phase out legacy squaring behavior. The "
                           "'legacy' setting will be removed in 0.26, and the "
                           "default setting will be changed to True. In 0.28, "
                           "'square_distances' will be removed altogether,"
                           "and distances will be squared by default. Set "
                           "'square_distances'=True to silence this warning."),
                          FutureWarning)
        if self.method == 'barnes_hut':
            X = self._validate_data(X, accept_sparse=['csr'],
                                    ensure_min_samples=2,
                                    dtype=[np.float32, np.float64])
        else:
            X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                    dtype=[np.float32, np.float64])
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(X, "TSNE.fit(). With metric='precomputed', X "
                                  "should contain positive distances.")

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    'or provide the dense distance matrix.')

        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric,
                                                   n_jobs=self.n_jobs)

            if np.any(distances < 0):
                raise ValueError("All distances should be positive, the "
                                 "metric given is not correct")

            if self.metric != "euclidean" and self.square_distances is True:
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors..."
                      .format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=self.n_jobs,
                                   n_neighbors=n_neighbors,
                                   metric=self.metric)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode='distance')
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples "
                      "in {:.3f}s...".format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.square_distances is True or self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity,
                                        self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        """
        self.fit_transform(X)
        return self


class JTSNEBASE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.
    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].
    Read more in the :ref:`User Guide <t_sne>`.
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    metric : string or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    init : string or numpy array, default="random"
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term: `Glossary <random_state>`.
    method : string, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.
        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.
    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.22
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from hoan.manifold import JTSNEBASE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = JTSNEBASE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    References
    ----------
    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/
    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
    """
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 n_jobs=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

    def _fit(self, X, alpha, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.method == 'barnes_hut':
            if type(X)==dict:
                for item in X:
                    X[item] = self._validate_data(X[item], accept_sparse=['csr'],
                                            ensure_min_samples=2,
                                            dtype=[np.float32, np.float64])
        else:
            if type(X)==dict:
                for item in X:
                    X[item] = self._validate_data(X[item], accept_sparse=['csr', 'csc', 'coo'],
                                    dtype=[np.float32, np.float64])
            else:
                X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                    dtype=[np.float32, np.float64])
        if self.metric == "precomputed":
            print("No implementation")

        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        if type(X)==dict:
            n_samples = next(iter(X.values())).shape[0]
        else:
            n_samples = X.shape[0]

        neighbors_nn = None
        distances = [None]*len(X)
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.

            if self.metric == "precomputed":
                print("No implementation")
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                for index, key in enumerate(X):
                    if self.metric == "euclidean":
                        distances[index] = pairwise_distances(X[key], metric=self.metric,
                                                       squared=True)
                    else:
                        distances[index] = pairwise_distances(X[key], metric=self.metric,
                                                       n_jobs=self.n_jobs)

                    if np.any(distances[index] < 0):
                        raise ValueError("All distances should be positive, the "
                                         "metric given is not correct")

            # compute the joint probability distribution for the input space
            P = alpha[0] * _joint_probabilities(distances[0], self.perplexity, self.verbose)
            for it in range(1, len(X)):
                P += alpha[it] * _joint_probabilities(distances[it], self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            # n_samples = next(iter(X.values())).shape[0]
            n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors..."
                      .format(n_neighbors))

            # Find the nearest neighbors for every point
            t0 = time()
            knn = [None]*len(X)
            distances_nn = [None]*len(X)
            for index, key in enumerate(X):
                knn[index] = NearestNeighbors(algorithm='auto', n_jobs=self.n_jobs, n_neighbors=n_neighbors, metric=self.metric)
                knn[index].fit(X[key])
                distances_nn[index] = knn[index].kneighbors_graph(mode='distance')

            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples "
                      "in {:.3f}s...".format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                for it in range(len(X)):
                    distances_nn[it].data **= 2

            # compute the joint probability distribution for the input space
            Pvec = [None]*len(X)
            Pvec[0] = _joint_probabilities_nn(distances_nn[0], self.perplexity,
                                        self.verbose)
            P = alpha[0] * Pvec[0]
            for it in range(1, len(X)):
                Pvec[it] = _joint_probabilities_nn(distances_nn[it], self.perplexity, self.verbose)
                P += alpha[it] * Pvec[it]

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(next(iter(X.values()))).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        res = self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

        # start = time()
        self.cross_entropy = [None]*len(X)
        for it in range(len(X)):
            self.cross_entropy[it], grad =  _kl_divergence_bh(res, Pvec[it], degrees_of_freedom, n_samples, self.n_components,
                      angle=self.angle, skip_num_points=0, verbose=False, compute_error=True, num_threads=1)
        # print("Time for computing entropy: ", time() - start)
        return res

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, alpha, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # print("Joint tSNE")
        embedding = self._fit(X, alpha)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, alpha, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        """
        self.fit_transform(X, alpha)
        return self


class JTSNE(JTSNEBASE):
    def fit_transform(self, X, method = 'uniform', max_iter = 10, _lambda = 5, y = None):
        """ Fit X into an embedded space and return that transformed output.
        Parameters
        ----------
        X: array, shape(n_samples, n_features)
        y: Ignored
        Returns 
        -------
        X_new: 
        """
        #print("JVis version 0.05")
        self.obj_value = 0 # store objective value
        if method == 'uniform':
            alpha = [1/len(X)]*len(X)
            embedding = self._fit(X, alpha)
            self.embedding_ = embedding 
            self.alpha = np.array(alpha)
            return self.embedding_ 
        elif method == "random-restart":
            alpha = [1/len(X)]*len(X)
            # weight = np.array([0]*len(X))
            for it in range(max_iter):
                embedding = self._fit(X, alpha)
                entropies = np.array(self.cross_entropy)
                alpha_np = np.array(alpha)
                self.obj_value = np.inner(entropies, alpha_np) + _lambda * np.inner(alpha_np, np.log(alpha_np))
                print("obj1 = ", self.obj_value)
                alpha = np.exp(-entropies/_lambda - 1)
                alpha = alpha/np.sum(alpha)
                self.obj_value = np.inner(entropies, alpha_np) + _lambda * np.inner(alpha_np, np.log(alpha_np))
                print("obj2 = ", self.obj_value)
                print(alpha)
            self.alpha = alpha
            self.embedding_ = embedding 
            return self.embedding_
        else:
            alpha = [1/len(X)]*len(X)
            self.init = self._fit(X, alpha)
            # weight = np.array([0]*len(X))
            for it in range(max_iter):
                # start_ = time()
                embedding = self._fit(X, alpha)
                entropies = np.array(self.cross_entropy)
                alpha_np = np.array(alpha)
                self.obj_value = np.inner(entropies, alpha_np) + _lambda * np.inner(alpha_np, np.log(alpha_np))
                # print(self.obj_value)
                # print("Total time = ", time() - start_)
                entropies = np.array(self.cross_entropy)
                self.init = embedding
                alpha = np.exp(-entropies/_lambda - 1)
                alpha = alpha/np.sum(alpha)
                alpha_np = np.array(alpha)
                self.obj_value = np.inner(entropies, alpha_np) + _lambda * np.inner(alpha_np, np.log(alpha_np))
            self.alpha = alpha
            self.embedding_ = embedding 
            return self.embedding_



#validation#####################

@numba.njit()
def trustworthiness_vector_bulk(
    indices_source, indices_embedded, max_k
):  # pragma: no cover

    n_samples = indices_embedded.shape[0]
    trustworthiness = np.zeros(max_k + 1, dtype=np.float64)

    for i in range(n_samples):
        for j in range(max_k):

            rank = 0
            while indices_source[i, rank] != indices_embedded[i, j]:
                rank += 1

            for k in range(j + 1, max_k + 1):
                if rank > k:
                    trustworthiness[k] += rank - k

    for k in range(1, max_k + 1):
        trustworthiness[k] = 1.0 - trustworthiness[k] * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )

    return trustworthiness


def make_trustworthiness_calculator(metric):  # pragma: no cover
    @numba.njit(parallel=True)
    def trustworthiness_vector_lowmem(source, indices_embedded, max_k):

        n_samples = indices_embedded.shape[0]
        trustworthiness = np.zeros(max_k + 1, dtype=np.float64)
        dist_vector = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):

            for j in numba.prange(n_samples):
                dist_vector[j] = metric(source[i], source[j])

            indices_source = np.argsort(dist_vector)

            for j in range(max_k):

                rank = 0
                while indices_source[rank] != indices_embedded[i, j]:
                    rank += 1

                for k in range(j + 1, max_k + 1):
                    if rank > k:
                        trustworthiness[k] += rank - k

        for k in range(1, max_k + 1):
            trustworthiness[k] = 1.0 - trustworthiness[k] * (
                2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
            )

        trustworthiness[0] = 1.0

        return trustworthiness

    return trustworthiness_vector_lowmem


def trustworthiness_vector(
    source, embedding, max_k, metric="euclidean"
):  # pragma: no cover
    tree = KDTree(embedding, metric=metric)
    indices_embedded = tree.query(embedding, k=max_k, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]

    dist = named_distances[metric]

    vec_calculator = make_trustworthiness_calculator(dist)

    result = vec_calculator(source, indices_embedded, max_k)

    return result



#%%

def JvisPy(datlist, 
           n_components, 
           metric, 
           random_state, 
           Lambda, 
           
           perplexity, 
           
           n_neighbors, 
           min_dist): 
    
    data = datlist
    
    n_components = int(n_components)
    metric = str(metric)
    
    if random_state is not None:
        random_state = int(random_state)
    
    Lambda = int(Lambda)
    
    perplexity = int(perplexity)
    n_neighbors = int(n_neighbors)
    min_dist = float(min_dist)
        
    
    # Run joint TSNE of the two "random" modalities.
    embedding_jtsne = JTSNE(n_components = n_components, 
                            metric = metric, 
                            random_state = random_state, 
                            perplexity = perplexity)
    
    
    embedding_jtsne = embedding_jtsne.fit_transform(X = data, _lambda = Lambda)

    # Run joint UMAP of the two "random" modalities.
    embedding_jumap = JUMAP(n_components = n_components, 
                            metric = metric, 
                            random_state = random_state, 
                            n_neighbors = n_neighbors, 
                            min_dist = min_dist)
    
    embedding_jumap = embedding_jumap.fit_transform(X = data, ld = Lambda)
    
    
    res = {'JSNE': embedding_jtsne, 'JUMAP': embedding_jumap}
    
    return res







