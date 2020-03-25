import numpy as np
import tqdm
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import pandas as pd
import robustats as rs


from bsz.utils import (
    bsplitz_method,
    _classification_summary_vector,
    fast_gini_improvements,
    fast_entropy_improvements,
    mae_improvement,
)


IMPROVEMENTS = {
    "gini": fast_gini_improvements,
    "entropy": fast_entropy_improvements,
    "mae": mae_improvement,
}


class NominalSplitter(object):
    def __init__(
        self,
        improvement_measure=fast_gini_improvements,
        random_state=None,
        num_samples=5000,
    ):
        self.improvement_measure = improvement_measure
        self.vec = None
        self.threshold = None
        self.improvement = None
        self.random_state = random_state
        self.num_samples = num_samples

    def cal_improvements(self, xi, y_high, y_uniques=None):
        np.random.seed(self.random_state)
        d = _classification_summary_vector(y_high)
        # self.vec = preprocessing.OneHotEncoder()
        self.vec = preprocessing.OneHotEncoder(handle_unknown="ignore")
        xi_high_d = self.vec.fit_transform(xi[:, np.newaxis])
        gs = xi_high_d.T.dot(y_high)

        pis, indices = bsplitz_method(gs, self.num_samples)
        # print('sampled pis', self.num_samples,
        #      pis.shape, indices.shape, np.unique(xi).shape, xi.shape)
        return np.nan_to_num(self.improvement_measure(pis, d, y_uniques)), indices

    def find_best(self, xi, y_high, y_uniques=None):
        improvements, indices = self.cal_improvements(xi, y_high, y_uniques)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        encoded = self.vec.transform(xi[:, np.newaxis])
        return encoded.dot(self.threshold) <= 0


class NumericalSplitter(NominalSplitter):
    def cal_improvements(self, orders, y_high, y_uniques=None):
        d = y_high.sum(axis=0).ravel()
        pis = np.cumsum(y_high[orders], axis=0)
        return np.nan_to_num(self.improvement_measure(pis, d, y_uniques))

    def find_best(self, xi, y_high, y_uniques=None):
        unique, counts = np.unique(xi, return_counts=True)
        valid_index = np.cumsum(counts) - 1
        orders = np.argsort(xi)
        indices = xi[orders]
        improvements = self.cal_improvements(orders, y_high, y_uniques)
        best_valid_index = np.argmax(improvements[valid_index])
        best_index = valid_index[best_valid_index]
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        return xi <= self.threshold


# class NumericalMedianSplitter(NumericalSplitter):
#     def cal_improvements(self, orders, y_high, y_uniques):
#         d = y_high.sum(axis=0).ravel()
#         pis = np.cumsum(y_high[orders], axis=0)
#         qis = d-pis
#         old_medians = mae([d], y_uniques)[0]
#         left_medians = mae(pis, y_uniques)
#         right_medians = mae(qis, y_uniques)
#         return old_medians - right_medians - left_medians
#
#     def find_best(self, xi, y_high, y_uniques):
#         unique, counts = np.unique(xi, return_counts=True)
#         valid_index = np.cumsum(counts) - 1
#         orders = np.argsort(xi)
#         indices = xi[orders]
#         improvements = self.cal_improvements(orders, y_high, y_uniques)
#         best_valid_index = np.argmax(improvements[valid_index])
#         best_index = valid_index[best_valid_index]
#         self.improvement = improvements[best_index]
#         self.threshold = indices[best_index]
#         return self.improvement


# class NominalMedianSplitter(NominalSplitter):
#     def cal_improvements(self, xi, y_high, y_uniques):
#         d = y_high.sum(axis=0).ravel()
#         self.vec = preprocessing.OneHotEncoder(handle_unknown="ignore")
#         xi_high_d = self.vec.fit_transform(xi[:, np.newaxis])
#         gs = xi_high_d.T.dot(y_high)
#
#         pis, indices = bsplitz_method(gs, self.num_samples)
#         qis = d-pis
#         old_medians = mae([d], y_uniques)[0]
#         left_medians = mae(pis, y_uniques)
#         right_medians = mae(qis, y_uniques)
#         return old_medians - right_medians - left_medians
#
#     def find_best(self, xi, y_high, y_uniques):
#         improvements, indices = self.cal_improvements(xi, y_high, y_uniques)
#         best_index = np.argmax(improvements)
#         self.improvement = improvements[best_index]
#         self.threshold = indices[best_index]
#         return self.improvement


class VecNumericalSplitter(NominalSplitter):
    def cal_improvements(self, orders, y_high):
        d = y_high.sum(axis=0).ravel()
        d = np.hstack([d, d.sum(keepdims=True)])
        pis = np.cumsum(y_high[orders], axis=0)
        pis = np.concatenate([pis, pis.sum(axis=2, keepdims=True)], axis=2)
        # import ipdb; ipdb.set_trace()
        return np.nan_to_num(self.improvement_measure(pis, d))

    def find_best(self, x, y_high):
        orders = np.argsort(x)
        indices = x[orders]
        improvements = self.cal_improvements(orders, y_high)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index]
        return self.improvement


def data_to_probs(y_high_d):
    return np.nan_to_num(y_high_d.sum(0) / np.sum(y_high_d)).ravel()


class BsplitZClassifier(BaseEstimator, ClassifierMixin):
    """
    BSplitZ Decision Stump classifier supports native nominal features
    """

    def __init__(
        self,
        nominal_cols=None,
        criteria="gini",
        verbose=False,
        random_state=None,
        num_samples=5000,
    ):
        """

        :param nominal_cols: list of cols that is nominal
        :param criteria: splitting criteria.
        """
        self.nominal_cols_ = nominal_cols or []
        self.criteria = criteria
        self.res_ = None
        self.verbose = verbose
        self.random_state_ = random_state
        self.num_samples = num_samples

    def _fit_data(self, X, y_high_d, y_uniques):
        for i in tqdm.tqdm(range(X.shape[1]), disable=not self.verbose):
            xi = X[:, i]
            if i in self.nominal_cols_:
                splitter = NominalSplitter(
                    IMPROVEMENTS[self.criteria],
                    random_state=self.random_state_,
                    num_samples=self.num_samples,
                )
            else:
                splitter = NumericalSplitter(
                    IMPROVEMENTS[self.criteria], random_state=self.random_state_
                )
            improvement = splitter.find_best(xi, y_high_d, y_uniques)
            if improvement >= self.improvement_:
                self.improvement_ = improvement
                self.best_feature_ = i
                self.splitter_ = splitter

    def fit(self, X, y, sample_weight=None):
        self.improvement_ = -np.inf
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        # X, y = check_X_y(X, y)
        one = preprocessing.OneHotEncoder(sparse=False)
        y_high_d = one.fit_transform(y[:, np.newaxis])
        y_uniques = one.categories_[0]

        if sample_weight is not None:
            weighted_y_high = y_high_d.multiply(sample_weight[:, np.newaxis])
        else:
            weighted_y_high = y_high_d

        self._fit_data(X, weighted_y_high, y_uniques)
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
        return {
            "nominal_cols": self.nominal_cols_,
            "criteria": self.criteria,
            "random_state": self.random_state_,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class BSplitZRegressor(BsplitZClassifier):
    def fit(self, X, y, sample_weight=None):
        self.improvement_ = -np.inf
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        # X, y = check_X_y(X, y)
        one = preprocessing.OneHotEncoder(sparse=False)
        y_high_d = one.fit_transform(y[:, np.newaxis])
        y_uniques = one.categories_[0]

        if sample_weight is not None:
            weighted_y_high = y_high_d.multiply(sample_weight[:, np.newaxis])
        else:
            weighted_y_high = y_high_d

        self._fit_data(X, weighted_y_high, y_uniques)

        mask = self.splitter_.split(X[:, self.best_feature_])
        self.predictions_ = np.array(
            [
                np.median(
                    np.repeat(
                        y_uniques, np.array(y_high_d[mask].sum(axis=0), dtype=np.int32)
                    )
                ),
                np.median(
                    np.repeat(
                        y_uniques, np.array(y_high_d[~mask].sum(axis=0), dtype=np.int32)
                    )
                ),
            ]
        )
        return self

    def predict(self, x):
        check_is_fitted(self, ["best_feature_", "splitter_", "predictions_"])
        if isinstance(x, pd.DataFrame):
            x = x.values
        mask = self.splitter_.split(x[:, self.best_feature_])
        return self.predictions_[np.where(mask, 0, 1)]


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
