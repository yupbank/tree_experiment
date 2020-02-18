import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
from bsz.bsplitz import NumericalSplitter, NominalSplitter
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def classification_data():
    return make_classification()


def _fit_decision_tree_classifier(x, y):
    return DecisionTreeClassifier(max_depth=1).fit(x[:, np.newaxis], y)


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
    clf = _fit_decision_tree_classifier(x[:, col], y)
    tree_improvement = _tree_classifier_impurity(clf)
    np.testing.assert_almost_equal(np.max(improvements), tree_improvement)


def test_numerical_splitter_splits_correctly(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    numerical_splitter = NumericalSplitter()
    col = 4
    numerical_splitter.find_best(x[:, col], y_high_d)
    clf = _fit_decision_tree_classifier(x[:, col], y)

    np.testing.assert_equal(
        x[:, col] <= clf.tree_.threshold[0], x[:, col] <= numerical_splitter.threshold
    )

    np.testing.assert_equal(
        x[:, col] <= clf.tree_.threshold[0], numerical_splitter.split(x[:, col])
    )


def test_nominal_splitter_improvements(classification_data):
    x, y = classification_data
    y_high_d = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    nominal_splitter = NominalSplitter()
    col = 4
    xi = x[:, col]
    c_xi = np.digitize(xi, np.histogram(xi, bins=10)[1])
    best_improvement = nominal_splitter.find_best(c_xi, y_high_d)
    clf = _fit_decision_tree_classifier(c_xi, y)
    improvement_baseline = _tree_classifier_impurity(clf)

    assert best_improvement >= improvement_baseline
