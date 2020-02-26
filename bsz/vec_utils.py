import numpy as np
from functools import partial


def r_log_r(r):
    return r * np.log2(r)


def homo_to_euclidean(vecs):
    return vecs[:, :-1, :] / vecs[:, -1:, :]


def vec_gini(vecs):
    vecs = homo_to_euclidean(vecs)
    values = np.square(vecs).sum(axis=1)
    return np.ones_like(values) - values


def vec_entropy(vecs):
    vecs = homo_to_euclidean(vecs)
    return -r_log_r(vecs).sum(axis=1)


def vec_variance(vecs):
    vecs = homo_to_euclidean(vecs)
    return -np.square(vecs).sum(axis=1)


def skewness_varaince(vecs):
    vecs = homo_to_euclidean(vecs)
    return 2 * vecs[:, 0, :] ** 3 - 3 * vecs[:, 0, :] * vecs[:, 1, :]


def _impurity(measure, ps, d):
    qs = d - ps
    vecs = np.stack([ps, qs], axis=2)
    lr_sizes = vecs[:, -1, :]
    improvements = measure(vecs)
    base_improvement = measure(d[np.newaxis, :, np.newaxis]).ravel()
    return base_improvement - np.einsum("ij,ij->i", lr_sizes, improvements) / d[-1]


gini_impurity = partial(_impurity, vec_gini)
entropy_impurity = partial(_impurity, vec_entropy)
variance_impurity = partial(_impurity, vec_variance)
skewness_impurity = partial(_impurity, skewness_varaince)
