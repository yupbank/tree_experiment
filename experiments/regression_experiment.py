import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.datasets import load_boston
import scipy.stats as stats
import itertools
from bsz.utils import (
    bsplitz_method,
    fast_skewness_improvements,
    fast_variance_improvements,
)
from bsz.cube_to_zonotope import enumerate_all_points, vertex_points


def report_regression_comparison(
    mean_encoding,
    generators,
    d,
    measure=fast_skewness_improvements,
    measure_name="skewness",
):
    potentials = np.cumsum(generators[np.argsort(mean_encoding)], axis=0)
    res = measure(potentials, d)

    all_candidates = enumerate_all_points(generators)
    new_res = measure(all_candidates, d)

    another_potentials, _ = bsplitz_method(generators)
    similar_res = measure(another_potentials, d)

    vertex_potentials = vertex_points(all_candidates)
    vertex_res = measure(vertex_potentials, d)

    print("best %s improvement from mean encoding" % measure_name, np.max(res))
    print("best %s improvement from greedy enumerating" % measure_name, np.max(new_res))
    print(
        "best %s improvement from smart enumerating" % measure_name, np.max(similar_res)
    )
    print(
        "best %s improvement from vertext of greedy enumerating" % measure_name,
        np.max(vertex_res),
    )


def main():
    data = load_boston()
    x = data["data"][:, 8]
    y = data["target"]
    d = np.array([y.sum(), (y ** 2).sum(), y.shape[0]])
    hx = OneHotEncoder(sparse=False, categories="auto").fit_transform(x[:, np.newaxis])

    generators = np.vstack([hx.T.dot(y), hx.T.dot(y ** 2), hx.sum(axis=0)]).T

    mean_encoding = generators[:, 0] / generators[:, -1]

    report_regression_comparison(mean_encoding, generators, d)
    report_regression_comparison(
        mean_encoding, generators, d, fast_variance_improvements, "variance"
    )


if __name__ == "__main__":
    main()
