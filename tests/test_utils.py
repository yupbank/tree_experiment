import numpy as np
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
def regression_target_split(regression_data):
    x, y = regression_data
    index = 4
    return y, index


@pytest.fixture
def classification_target_split(classification_data):
    x, y, y_high_d = classification_data
    index = 4
    return y_high_d, index


@pytest.mark.parametrize(
    "simple_method,vec_method",
    [
        (variance_improvement, fast_variance_improvements),
        (skewness_improvement, fast_skewness_improvements),
    ],
)
def test_regression_match(regression_target_split, simple_method, vec_method):
    data, index = regression_target_split
    res = simple_method(data, index)

    d = _regression_summary_vector(data)
    p = _regression_summary_vector(data[:index])

    f_res = vec_method(np.array([p, p]), d)
    np.testing.assert_allclose(res, f_res[0])
    np.testing.assert_allclose(f_res[0], f_res[1])


@pytest.mark.parametrize(
    "simple_method,vec_method",
    [
        (gini_improvement, fast_gini_improvements),
        (entropy_improvement, fast_entropy_improvements),
    ],
)
def test_classification_math(classification_target_split, simple_method, vec_method):
    data, index = classification_target_split
    res = simple_method(data, index)
    d = _classification_summary_vector(data)
    p = _classification_summary_vector(data[:index])
    f_res = vec_method(np.array([p, p]), d)
    np.testing.assert_allclose(res, f_res[0])
    np.testing.assert_allclose(f_res[0], f_res[1])
