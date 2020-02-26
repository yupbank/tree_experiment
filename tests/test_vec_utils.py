import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_classification
from bsz.utils import (
    fast_gini_improvements,
    fast_variance_improvements,
    fast_skewness_improvements,
    skewness_improvement,
    _regression_summary_vector,
    gini_improvement,
    _classification_summary_vector,
    entropy_improvement,
    fast_entropy_improvements,
)
from bsz import vec_utils
import pytest


@pytest.fixture
def regression_target_split():
    data = np.random.randn(20)
    index = 4
    return data, index


@pytest.fixture
def classification_target_split():
    x, data = make_classification()
    index = 4
    return data, index


def test_fast_variance_the_same_with_variance(regression_target_split):
    data, index = regression_target_split

    d = _regression_summary_vector(data)[[0, 2]]
    p = _regression_summary_vector(data[:index])[[0, 2]]

    f_res = fast_variance_improvements(np.array([p, p]), d)
    v_res = vec_utils.variance_impurity(np.array([p, p]), d)
    np.testing.assert_allclose(f_res, v_res)


def test_fast_skewness_the_same_with_skewness(regression_target_split):
    data, index = regression_target_split

    d = _regression_summary_vector(data)
    p = _regression_summary_vector(data[:index])

    f_res = fast_skewness_improvements(np.array([p, p]), d)
    v_res = vec_utils.skewness_impurity(np.array([p, p]), d)

    np.testing.assert_allclose(v_res, f_res)

    np.testing.assert_allclose(f_res[0], f_res[1])


def test_fast_gini_the_same_with_gini(classification_target_split):
    y, index = classification_target_split

    data = preprocessing.OneHotEncoder(sparse=False, categories="auto").fit_transform(
        y[:, np.newaxis]
    )

    d = _classification_summary_vector(data)
    p = _classification_summary_vector(data[:index])
    p_new = np.hstack([p, p.sum()])
    d_new = np.hstack([d, d.sum()])

    f_res = fast_gini_improvements(np.array([p, p]), d)
    v_res = vec_utils.gini_impurity(np.array([p_new, p_new]), d_new)

    np.testing.assert_allclose(v_res, f_res)

    np.testing.assert_allclose(f_res[0], f_res[1])


def test_fast_entropy_the_same_with_entropy(classification_target_split):
    y, index = classification_target_split

    data = preprocessing.OneHotEncoder(sparse=False, categories="auto").fit_transform(
        y[:, np.newaxis]
    )

    d = _classification_summary_vector(data)
    p = _classification_summary_vector(data[:index])
    p_new = np.hstack([p, p.sum()])
    d_new = np.hstack([d, d.sum()])

    f_res = fast_entropy_improvements(np.array([p, p]), d)
    v_res = vec_utils.entropy_impurity(np.array([p_new, p_new]), d_new)

    np.testing.assert_allclose(v_res, f_res)

    np.testing.assert_allclose(f_res[0], f_res[1])
