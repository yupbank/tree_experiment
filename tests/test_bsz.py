import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
from bsz.bsplitz import (
    NumericalSplitter,
    NominalSplitter,
    BsplitZClassifier,
    VecNumericalSplitter,
    mae_improvement,
)
from sklearn.tree import DecisionTreeClassifier
from bsz import vec_utils


@pytest.fixture
def classification_data():
    return (
        np.array(
            [
                [1.27218253, 0.5601962, 3.16651161, 0.31530069, -0.31402958],
                [0.09277657, 0.70245147, -1.17064371, 1.5789989, -2.15371559],
                [1.56782099, -0.78768125, 0.2858491, 1.01473497, 1.64521412],
                [-0.95046474, -1.44892416, -1.9103314, -0.62311653, 0.30237404],
                [-0.20403424, -1.80838061, 0.70055809, -1.56507989, -1.5060645],
                [-1.22329363, 0.34663416, 0.6943241, -0.07918017, -0.39035781],
            ]
        ),
        np.array([0, 0, 0, 1, 1, 0]),
    )


def _fit_decision_tree_classifier(x, y):
    return DecisionTreeClassifier(max_depth=1).fit(x, y)


def _tree_classifier_impurity(clf):
    return np.dot(
        np.array([1, -1, -1]),
        clf.tree_.impurity * clf.tree_.n_node_samples / clf.tree_.n_node_samples[0],
    )


def test_numerical_splitter_improvements(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    numerical_splitter = NumericalSplitter()
    col = 4
    orders = np.argsort(x[:, col])
    improvements = numerical_splitter.cal_improvements(orders, y_high_d)
    clf = _fit_decision_tree_classifier(x[:, col][:, np.newaxis], y)
    tree_improvement = _tree_classifier_impurity(clf)
    np.testing.assert_almost_equal(np.max(improvements), tree_improvement)


def test_vec_numerical_splitter_improvemtns(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    splitter = VecNumericalSplitter(vec_utils.mat_gini_impurity)
    numerical_splitter = NumericalSplitter()
    col = 4
    orders = np.argsort(x, axis=0)
    improvements = numerical_splitter.cal_improvements(orders[:, col], y_high_d)
    massive_improvements = splitter.cal_improvements(orders, y_high_d)
    np.testing.assert_almost_equal(improvements, massive_improvements[:, col])


def test_numerical_splitter_splits_correctly(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    numerical_splitter = NumericalSplitter()
    col = 4
    numerical_splitter.find_best(x[:, col], y_high_d)
    clf = _fit_decision_tree_classifier(x[:, col][:, np.newaxis], y)
    np.testing.assert_equal(
        x[:, col] < clf.tree_.threshold[0], x[:, col] <= numerical_splitter.threshold
    )

    np.testing.assert_equal(
        x[:, col] < clf.tree_.threshold[0], numerical_splitter.split(x[:, col])
    )


def test_nominal_splitter_improvements(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    nominal_splitter = NominalSplitter()
    col = 4
    xi = x[:, col]
    c_xi = np.digitize(xi, np.histogram(xi, bins=10)[1])
    best_improvement = nominal_splitter.find_best(c_xi, y_high_d)
    clf = _fit_decision_tree_classifier(c_xi[:, np.newaxis], y)
    improvement_baseline = _tree_classifier_impurity(clf)

    assert best_improvement >= improvement_baseline


def test_nominal_splitter_splits_correctly(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    nominal_splitter = NominalSplitter()
    col = 4
    xi = x[:, col]
    c_xi = np.digitize(xi, np.histogram(xi, bins=10)[1])
    nominal_splitter.find_best(c_xi, y_high_d)
    np.testing.assert_equal(
        nominal_splitter.vec.transform(c_xi[:, np.newaxis]).dot(
            nominal_splitter.threshold
        )
        <= 0,
        nominal_splitter.split(c_xi),
    )


def test_numerical_splitter_improvements():
    from bsz.utils import fast_entropy_improvements

    y_high_d = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    orders = np.array([3, 1, 2, 5, 4, 0])
    splitter = NumericalSplitter(fast_entropy_improvements)
    improvements = splitter.cal_improvements(orders, y_high_d)
    assert not np.all(improvements == 0)


@pytest.mark.parametrize("criteria", ["gini", "entropy"])
def test_bsz_default_classifier_similar_to_decision_tree(criteria, classification_data):
    x, y = classification_data
    bsz_clf = BsplitZClassifier(criteria=criteria).fit(x, y)
    clf = DecisionTreeClassifier(max_depth=1, criterion=criteria).fit(x, y)
    tree_improvement = _tree_classifier_impurity(clf)
    np.testing.assert_almost_equal(bsz_clf.improvement_, tree_improvement)

    np.testing.assert_almost_equal(bsz_clf.predict(x), clf.predict(x))

    np.testing.assert_almost_equal(
        bsz_clf.predict_proba(x), clf.predict_proba(x), decimal=1
    )


def test_numerical_median_splitter():
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
    nms = NumericalSplitter(mae_improvement)
    actual = nms.find_best(x, y_high, y_uniques)
    from sklearn.tree import DecisionTreeRegressor

    clf = DecisionTreeRegressor(criterion="mae", max_depth=1)
    clf.fit(x[:, np.newaxis], y)
    expected = np.dot(clf.tree_.impurity * clf.tree_.n_node_samples, [1, -1, -1])
    np.testing.assert_almost_equal(actual, expected)
