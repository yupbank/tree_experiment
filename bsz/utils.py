import numpy as np
from bsz.cube_to_zonotope import sample_zonotope_vertices, aggregate_generators_by_direction
from sklearn import preprocessing
from functools import partial


def bsplitz_method(gs):
    processed_gs, inverse_index = aggregate_generators_by_direction(gs)
    ps, indices = sample_zonotope_vertices(processed_gs, 5000, 1)
    full_indices = indices[:, inverse_index]
    return ps, full_indices


def numerical_split(feature_vector, threshold_vector, vectorizer):
    encoded = np.vstack([feature_vector, np.ones_like(feature_vector)]).T
    return encoded.dot(threshold_vector) >= 0


def nominal_split(feature_vector, threshold_vector, vectorizer):
    encoded = vectorizer.transform(feature_vector[:, np.newaxis])
    return encoded.dot(threshold_vector) > 0


def split_data(feature_vector, threshold_vector, vectorizer, feature_type):
    if feature_type == 'numerical':
        return numerical_split(feature_vector, threshold_vector, vectorizer)
    elif feature_type == 'nominal':
        return nominal_split(feature_vector, threshold_vector, vectorizer)


def fast_gini_improvements(ps, d):
    dp = ps.dot(d)[:, np.newaxis]
    return (np.square((ps-dp)*d)/(dp*(1-dp))).sum(axis=1)


def _convex_improvements(measure, ps, d):
    overall_measure = measure(d[np.newaxis, :])
    dp = ps.dot(d)
    left_ratios = ps*d*np.reciprocal(dp[:, np.newaxis])
    left_measure = measure(left_ratios)
    right_ratios = (1-ps)*d*np.reciprocal(1-dp[:, np.newaxis])
    right_measure = measure(right_ratios)
    return overall_measure - dp*left_measure - (1-dp)*right_measure


def gini(ratios):
    return 1 - (ratios**2).sum(axis=1)


def entropy(ratios):
    return - (ratios*np.log(ratios)).sum(axis=1)


gini_improvements = partial(_convex_improvements, gini)
entropy_improvements = partial(_convex_improvements, entropy)


def variance_improvements(ps, d):
    constant = -(d[0]/d[-1])**2
    a = np.vstack([ps[:, -1], d[-1]-ps[:, -1]]).T
    b = np.vstack([ps[:, 0], d[0] - ps[:, 0]]).T
    e = (b/a) ** 2
    return np.nan_to_num(constant + np.einsum('ij,ij->i', a, e)/d[-1])