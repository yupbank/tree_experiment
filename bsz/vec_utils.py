import numpy as np
from functools import partial


def r_log_r(r):
    return np.log2(r ** r)


def matrix_homo_to_euclidean(mat):
    return mat[:, :, :-1, :] / mat[:, :, -1:, :]


def homo_to_euclidean(vecs):
    return vecs[:, :-1, :] / vecs[:, -1:, :]


def vec_gini(vecs):
    vecs = homo_to_euclidean(vecs)
    values = np.square(vecs).sum(axis=1)
    return np.ones_like(values) - values


def mat_gini(mat):
    mat = matrix_homo_to_euclidean(mat)
    values = np.square(mat).sum(axis=2)
    return np.ones_like(values) - values


def vec_entropy(vecs):
    vecs = homo_to_euclidean(vecs)
    return -r_log_r(vecs).sum(axis=1)


def mat_entropy(mat):
    mat = matrix_homo_to_euclidean(mat)
    return -r_log_r(mat).sum(axis=2)


def vec_variance(vecs):
    vecs = homo_to_euclidean(vecs)
    return -vecs[:, 0, :] ** 2


def mat_variance(mat):
    mat = matrix_homo_to_euclidean(mat)
    return -mat[:, :, 0, :] ** 2


def vec_skewness(vecs):
    vecs = homo_to_euclidean(vecs)
    return 2 * vecs[:, 0, :] ** 3 - 3 * vecs[:, 0, :] * vecs[:, 1, :]


def mat_skewness(mat):
    mat = matrix_homo_to_euclidean(mat)
    return 2 * mat[:, :, 0, :] ** 3 - 3 * mat[:, :, 0, :] * mat[:, :, 1, :]


def _vec_impurity(measure, ps, d):
    qs = d - ps
    vecs = np.stack([ps, qs], axis=2)
    lr_sizes = vecs[:, -1, :]
    improvements = measure(vecs)
    base_improvement = measure(d[np.newaxis, :, np.newaxis]).ravel()
    return np.nan_to_num(
        base_improvement - np.einsum("ij,ij->i", lr_sizes, improvements) / d[-1]
    )


def _mat_impurity(matrix_measure, ps, d):
    qs = d - ps
    mat = np.stack([ps, qs], axis=3)
    lr_sizes = mat[:, :, -1, :]
    improvements = matrix_measure(mat)
    base_improvement = matrix_measure(d[np.newaxis, np.newaxis, :, np.newaxis]).ravel()
    return np.nan_to_num(
        base_improvement - np.einsum("ijk,ijk->ij", lr_sizes, improvements) / d[-1]
    )


gini_impurity = partial(_vec_impurity, vec_gini)
entropy_impurity = partial(_vec_impurity, vec_entropy)
variance_impurity = partial(_vec_impurity, vec_variance)
skewness_impurity = partial(_vec_impurity, vec_skewness)

mat_gini_impurity = partial(_mat_impurity, mat_gini)
mat_entropy_impurity = partial(_mat_impurity, mat_entropy)
mat_variance_impurity = partial(_mat_impurity, mat_variance)
mat_skewness_impurity = partial(_mat_impurity, mat_skewness)
