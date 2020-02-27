import numpy as np

from bsz.utils import (
    fast_gini_improvements,
    fast_variance_improvements,
    fast_skewness_improvements,
    fast_entropy_improvements,
)
from bsz import vec_utils
import pytest


@pytest.mark.parametrize(
    "simple_method,vec_method,mat_method",
    [
        (
            fast_variance_improvements,
            vec_utils.variance_impurity,
            vec_utils.mat_variance_impurity,
        ),
        (
            fast_skewness_improvements,
            vec_utils.skewness_impurity,
            vec_utils.mat_skewness_impurity,
        ),
    ],
)
def test_regression_criteria(
    regression_ps_and_d, simple_method, vec_method, mat_method
):
    ps, d = regression_ps_and_d

    f_res = simple_method(ps, d)
    v_res = vec_method(ps, d)
    m_res = mat_method(np.array([ps, ps]), d)

    np.testing.assert_allclose(v_res, f_res)
    np.testing.assert_allclose(m_res[0], f_res)
    np.testing.assert_allclose(m_res[0], m_res[1])


@pytest.mark.parametrize(
    "simple_method,vec_method,mat_method",
    [
        (fast_gini_improvements, vec_utils.gini_impurity, vec_utils.mat_gini_impurity),
        (
            fast_entropy_improvements,
            vec_utils.entropy_impurity,
            vec_utils.mat_entropy_impurity,
        ),
    ],
)
def test_classification_criteria(
    classification_ps_and_d, simple_method, vec_method, mat_method
):
    ps, d = classification_ps_and_d
    ps_new = np.hstack([ps, ps.sum(axis=1, keepdims=True)])
    d_new = np.hstack([d, d.sum(keepdims=True)])

    f_res = simple_method(ps, d)
    v_res = vec_method(ps_new, d_new)
    m_res = mat_method(np.array([ps_new, ps_new]), d_new)

    np.testing.assert_allclose(v_res, f_res)
    np.testing.assert_allclose(m_res[0], f_res)

    np.testing.assert_allclose(m_res[0], m_res[1])
