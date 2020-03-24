from functools import partial

import numpy as np
import scipy.stats as stats

from bsz.cube_to_zonotope import (
    sample_zonotope_vertices,
    aggregate_generators_by_direction,
)


def bsplitz_method(gs, num_samples=5000):
    processed_gs, inverse_index = aggregate_generators_by_direction(gs)
    ps, indices = sample_zonotope_vertices(processed_gs, num_samples, 1)
    full_indices = indices[:, inverse_index]
    return ps, full_indices


def fast_skewness_improvements(ps, d):
    constant = 2 * (d[0] / d[-1]) ** 3 - 3 * (d[0] / d[-1] * d[1] / d[-1])
    a = np.vstack([ps[:, -1], d[-1] - ps[:, -1]]).T
    b = np.vstack([ps[:, 0], d[0] - ps[:, 0]]).T
    c = np.vstack([ps[:, 1], d[1] - ps[:, 1]]).T
    e = 3 * b / a * c / a - 2 * (b / a) ** 3
    return np.nan_to_num(constant + np.einsum("ij,ij->i", a, e) / d[-1])


def fast_variance_improvements(ps, d):
    N = d[-1]
    constant = -((d[0] / N) ** 2)
    a = np.vstack([ps[:, -1], N - ps[:, -1]]).T
    b = np.vstack([ps[:, 0], d[0] - ps[:, 0]]).T
    e = np.square(b / a)
    return np.nan_to_num(constant + np.einsum("ij,ij->i", a, e) / N)


def fast_gini_improvements(ps, d):
    N = d.sum()
    qs = d - ps
    constant = -np.square(d / N).sum()
    a = np.vstack([ps.sum(axis=1), N - ps.sum(axis=1)]).T
    e = np.vstack(
        [
            np.square(ps / np.sum(ps, axis=1, keepdims=True)).sum(axis=1),
            np.square(qs / np.sum(qs, axis=1, keepdims=True)).sum(axis=1),
        ]
    ).T
    return np.nan_to_num(constant + np.einsum("ij,ij->i", a, e) / N)


def _p_log_p(r):
    return np.log2(r ** r)


def fast_entropy_improvements(ps, d):
    N = d.sum()
    qs = d - ps
    constant = -(_p_log_p(d / N)).sum()
    a = np.vstack([ps.sum(axis=1), N - ps.sum(axis=1)]).T
    e = np.vstack(
        [
            _p_log_p(ps / np.sum(ps, axis=1, keepdims=True)).sum(axis=1),
            _p_log_p(qs / np.sum(qs, axis=1, keepdims=True)).sum(axis=1),
        ]
    ).T
    return np.nan_to_num(constant + np.einsum("ij,ij->i", a, e) / N)


def _regression_summary_vector(y):
    return np.array([y.sum(), (y ** 2).sum(), y.shape[0]])


def _classification_summary_vector(y):
    return y.sum(axis=0).ravel()


def _improvements(func, ys, index_l):
    length = float(ys.shape[0])
    if index_l == 0 or index_l == length:
        return 0.0
    return np.nan_to_num(
        func(ys)
        - index_l / length * func(ys[:index_l])
        - (1 - index_l / length) * func(ys[index_l:])
    )


def _gini(y):
    d = _classification_summary_vector(y)
    return 1 - np.square(d / np.sum(d, keepdims=True)).sum()


def _entropy(y):
    d = _classification_summary_vector(y)
    p = d / np.sum(d, keepdims=True)
    return -(_p_log_p(p)).sum()


variance_improvement = partial(_improvements, np.var)
skewness_improvement = partial(_improvements, lambda r: stats.moment(r, 3))
gini_improvement = partial(_improvements, _gini)
entropy_improvement = partial(_improvements, _entropy)
