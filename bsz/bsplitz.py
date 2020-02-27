import numpy as np
import tqdm
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import pandas as pd

from bsz.utils import (
    bsplitz_method,
    _classification_summary_vector,
    fast_gini_improvements,
    fast_entropy_improvements,
)

IMPROVEMENTS = {"gini": fast_gini_improvements, "entropy": fast_entropy_improvements}


class NominalSplitter(object):
    def __init__(self, improvement_measure=fast_gini_improvements):
        self.improvement_measure = improvement_measure
        self.vec = None
        self.threshold = None
        self.improvement = None

    def cal_improvements(self, xi, y_high):
        d = _classification_summary_vector(y_high)
        self.vec = preprocessing.OneHotEncoder()
        xi_high_d = self.vec.fit_transform(xi[:, np.newaxis])
        gs = xi_high_d.T.dot(y_high)

        pis, indices = bsplitz_method(gs)
        return np.nan_to_num(self.improvement_measure(pis, d)), indices

    def find_best(self, xi, y_high):
        improvements, indices = self.cal_improvements(xi, y_high)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        encoded = self.vec.transform(xi[:, np.newaxis])
        return encoded.dot(self.threshold) <= 0


class NumericalSplitter(NominalSplitter):
    def cal_improvements(self, orders, y_high):
        d = y_high.sum(axis=0).ravel()
        pis = np.cumsum(y_high[orders], axis=0)
        return np.nan_to_num(self.improvement_measure(pis, d))

    def find_best(self, xi, y_high):
        orders = np.argsort(xi)
        indices = xi[orders]
        improvements = self.cal_improvements(orders, y_high)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index + 1]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        return xi < self.threshold


class VecNumericalSplitter(NominalSplitter):
    def cal_improvements(self, orders, y_high):
        d = y_high.sum(axis=0).ravel()
        pis = np.cumsum(y_high[orders], axis=1)
        return np.nan_to_num(self.improvement_measure(pis, d))

    def find_best(self, x, y_high):
        orders = np.argsort(x)
        indices = x[orders]
        improvements = self.cal_improvements(orders, y_high)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index + 1]
        return self.improvement


def data_to_probs(y_high_d):
    return np.nan_to_num(y_high_d.sum(0) / np.sum(y_high_d)).ravel()


class BsplitZClassifier(BaseEstimator, ClassifierMixin):
    """
    BSplitZ Decision Stump classifier supports native nominal features
    """

    def __init__(self, nominal_cols=None, criteria="gini", verbose=False):
        """

        :param nominal_cols: list of columns that is nominal features, if not specified treat evey feature as numerical
        :param criteria: splitting criteria.
        """
        self.nominal_features_ = nominal_cols or []
        self.criteria = criteria
        self.res_ = None
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        self.improvement_ = -np.inf
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        X, y = check_X_y(X, y)
        one = preprocessing.OneHotEncoder(sparse=False)
        y_high_d = one.fit_transform(y[:, np.newaxis])

        if sample_weight is not None:
            weighted_y_high = y_high_d.multiply(sample_weight[:, np.newaxis])
        else:
            weighted_y_high = y_high_d

        for i in tqdm.tqdm(range(X.shape[1]), disable=not self.verbose):
            xi = X[:, i]
            if i in self.nominal_features_:
                splitter = NominalSplitter(IMPROVEMENTS[self.criteria])
            else:
                splitter = NumericalSplitter(IMPROVEMENTS[self.criteria])
            improvement = splitter.find_best(xi, weighted_y_high)
            if improvement >= self.improvement_:
                self.improvement_ = improvement
                self.best_feature_ = i
                self.splitter_ = splitter

        self.classes_ = one.categories_[0]
        mask = self.splitter_.split(X[:, self.best_feature_])
        self.classes_ = one.categories_[0]
        self.predictions_ = np.array(
            [data_to_probs(y_high_d[mask]), data_to_probs(y_high_d[~mask])]
        )
        return self

    def predict_proba(self, x):
        check_is_fitted(
            self, ["best_feature_", "splitter_", "predictions_", "classes_"]
        )
        if isinstance(x, pd.DataFrame):
            x = x.values
        mask = self.splitter_.split(x[:, self.best_feature_])
        return self.predictions_[np.where(mask, 0, 1)]

    def predict(self, x):
        prob = self.predict_proba(x)
        return self.classes_[np.argmax(prob, axis=1)]

    def get_params(self, deep=True):
        return {"nominal_cols": self.nominal_cols, "criteria": self.criteria}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


"""
if __name__ == "__main__":
    import openml

    dataset_meta_info = openml.datasets.get_dataset(1457, False)
    nominal_cols = ",".join(
        [
            str(key)
            for key, value in dataset_meta_info.features.items()
            if value.data_type == "nominal"
        ]
    )
    # clf = DecisionTreeClassifier(max_depth=1)
    clf = BsplitZClassifier(
        nominal_method="bsplitz",
        default_feature_type="nominal",
        non_default_feature_colmns=nominal_cols,
    )
    task = openml.tasks.get_task(56571)
    run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)

    # The run may be stored offline, and the flow will be stored along with it:
    # run.to_filesystem(directory='new_myrun')

    # They may be loaded and uploaded at a later time
    # run = openml.runs.OpenMLRun.from_filesystem(directory='new_myrun')
    print(run)
    run.publish()
"""
