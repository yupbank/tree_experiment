import numpy as np
import scipy.stats as stats
from bsz.utils import fast_gini_improvements, gini_improvements
import pytest


@pytest.fixture
def inputs():
    ps = np.random.randn(10, 5)
    return ps[:-1], ps[-1]


def test_fast_gini_the_same_with_gini(inputs):
    ps, d = inputs
    res = gini_improvements(ps, d)
    f_res = fast_gini_improvements(ps, d)
    np.testing.assert_allclose(
        res, f_res)
