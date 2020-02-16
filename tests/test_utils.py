import numpy as np
import scipy.stats as stats
from bsz.utils import fast_gini_improvements, gini_improvements, fast_variance_improvements, fast_skewness_improvements, variance_improvement, skewness_improvement, _summary_vector
import pytest


@pytest.fixture
def classification_inputs():
    ps = np.random.randn(10, 5)
    return ps[:-1], ps[-1]


def test_fast_gini_the_same_with_gini(classification_inputs):
    ps, d = classification_inputs
    res = gini_improvements(ps, d)
    f_res = fast_gini_improvements(ps, d)
    np.testing.assert_allclose(
        res, f_res)


@pytest.fixture
def regression_input():
    data = np.random.randn(20)
    index = 4
    return data, index


def test_fast_variance_the_same_with_variance(regression_input):
    data, index = regression_input
    res = variance_improvement(data, index)

    d = _summary_vector(data)
    p = _summary_vector(data[:index])

    f_res = fast_variance_improvements(np.array([p, p]), d)

    np.testing.assert_allclose(
        res, f_res[0])

    np.testing.assert_allclose(
        f_res[0], f_res[1])
