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
    mae,
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


def test_mae():
    y = np.array(
        [
            2.80000e-02,
            8.04000e-01,
            4.00000e-03,
            3.60000e03,
            3.60000e03,
            5.72000e-01,
            3.60000e03,
            1.28756e02,
            3.60000e03,
            1.90850e01,
        ]
    )
    x = np.array(
        [
            -1.54042965,
            -0.93462145,
            -0.53668933,
            -0.13294344,
            0.17720193,
            0.45695202,
            -0.80966124,
            -0.01791948,
            0.70969338,
            -0.69261156,
        ]
    )
    orders = np.array([0, 1, 6, 9, 2, 3, 7, 4, 5, 8])
    y_uniques = np.array(
        [
            4.00000e-03,
            2.80000e-02,
            5.72000e-01,
            8.04000e-01,
            1.90850e01,
            1.28756e02,
            3.60000e03,
        ]
    )
    y_high = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_almost_equal(np.sum(y_high * y_uniques, axis=1), y)
    pis = np.cumsum(y_high[orders], axis=0)
    d = pis[-1]
    actual_ae = mae(pis, y_uniques)
    ordered_y = y[orders]

    def raw_ae(y):
        return np.sum(np.abs(y - np.median(y)))

    expected_ae = [raw_ae(ordered_y[: i + 1]) for i in range(y.shape[0])]
    np.testing.assert_almost_equal(expected_ae, actual_ae)
    expected_ae = [raw_ae(ordered_y[i + 1 :]) for i in range(y.shape[0])]
    qis = d - pis
    actual_ae = mae(qis, y_uniques)
    np.testing.assert_almost_equal(expected_ae, actual_ae)
