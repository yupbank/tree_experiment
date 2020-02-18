import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_classification
from bsz.utils import (
    fast_gini_improvements,
    fast_variance_improvements,
    fast_skewness_improvements,
    variance_improvement,
    skewness_improvement,
    _regression_summary_vector,
    gini_improvement,
    _classification_summary_vector,
    entropy_improvement,
    fast_entropy_improvements,
)
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
    res = variance_improvement(data, index)

    d = _regression_summary_vector(data)
    p = _regression_summary_vector(data[:index])

    f_res = fast_variance_improvements(np.array([p, p]), d)

    np.testing.assert_allclose(res, f_res[0])

    np.testing.assert_allclose(f_res[0], f_res[1])


def test_fast_skewness_the_same_with_skewness(regression_target_split):
    data, index = regression_target_split
    res = skewness_improvement(data, index)

    d = _regression_summary_vector(data)
    p = _regression_summary_vector(data[:index])

    f_res = fast_skewness_improvements(np.array([p, p]), d)

    np.testing.assert_allclose(res, f_res[0])

    np.testing.assert_allclose(f_res[0], f_res[1])


def test_fast_gini_the_same_with_gini(classification_target_split):
    y, index = classification_target_split

    data = preprocessing.OneHotEncoder(sparse=False, categories="auto").fit_transform(
        y[:, np.newaxis]
    )

    res = gini_improvement(data, index)

    d = _classification_summary_vector(data)
    p = _classification_summary_vector(data[:index])

    f_res = fast_gini_improvements(np.array([p, p]), d)

    np.testing.assert_allclose(res, f_res[0])

    np.testing.assert_allclose(f_res[0], f_res[1])


def test_fast_entropy_the_same_with_entropy(classification_target_split):
    y, index = classification_target_split

    data = preprocessing.OneHotEncoder(sparse=False, categories="auto").fit_transform(
        y[:, np.newaxis]
    )

    res = entropy_improvement(data, index)

    d = _classification_summary_vector(data)
    p = _classification_summary_vector(data[:index])

    f_res = fast_entropy_improvements(np.array([p, p]), d)

    np.testing.assert_allclose(res, f_res[0])

    np.testing.assert_allclose(f_res[0], f_res[1])
