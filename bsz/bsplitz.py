import numpy as np
from sklearn import preprocessing
from bsz.utils import bsplitz_method, _classification_summary_vector, fast_gini_improvements, fast_entropy_improvements
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import tqdm

IMPROVEMENTS = {'gini': fast_gini_improvements,
                'entropy': fast_entropy_improvements}


class NominalSplitter(object):
    def __init__(self, improvement_measure=fast_gini_improvements):
        self.improvement_measure = improvement_measure
        self.vec = None
        self.threshold = None
        self.improvement = None

    def improvements(self, orders, y_high):
        return

    def find_best(self, xi, y_high):
        d = _classification_summary_vector(y_high)

        self.vec = preprocessing.OneHotEncoder()
        xi_high_d = self.vec.fit_transform(xi[:, np.newaxis])

        gs = xi_high_d.T.dot(y_high)
        pis, indices = bsplitz_method(gs)
        improvements = np.nan_to_num(self.improvement_measure(pis, d))
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = indices[best_index]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        encoded = self.vec.transform(xi[:, np.newaxis])
        return encoded.dot(self.threshold) <= 0


class NumericalSplitter(NominalSplitter):
    def improvements(self, orders, y_high):
        d = y_high.sum(axis=0).ravel()
        pis = np.cumsum(y_high[orders], axis=0)
        return np.nan_to_num(self.improvement_measure(pis, d))

    def find_best(self, xi, weighted_y_high):
        orders = np.argsort(xi)
        improvements = self.improvements(orders, weighted_y_high)
        best_index = np.argmax(improvements)
        self.improvement = improvements[best_index]
        self.threshold = xi[orders[best_index]]
        return self.improvement

    def split(self, xi):
        assert self.threshold is not None, "can't split a no fitted splitter"
        return xi <= self.threshold


def data_to_probs(y_high_d):
    return np.nan_to_num(y_high_d.sum(0) / np.sum(y_high_d)).A.ravel()


class BsplitZClassifier(BaseEstimator, ClassifierMixin):
    """
    BSplitZ Decision Stump classifier supports native nominal features
    """

    def __init__(self, nominal_cols='', criteria='gini'):
        """

        :param nominal_cols: comma separated columns of nominal features, if not specified treat evey feature as numerical
        :param criteria: splitting criteria.
        """
        self.nominal_cols = nominal_cols
        self.criteria = criteria
        self.res_ = None

    def get_nominal_features(self):
        return [int(i) for i in filter(
            lambda r: r.strip(), self.nominal_cols.split(','))]

    def fit(self, X, y, sample_weight=None):
        self.res_ = {'improvement': -np.inf}
        nominal_features = self.get_nominal_features()

        X, y = check_X_y(X, y)

        one = preprocessing.OneHotEncoder()
        y_high_d = one.fit_transform(y[:, np.newaxis])

        if sample_weight is not None:
            weighted_y_high = y_high_d.multiply(sample_weight[:, np.newaxis])
        else:
            weighted_y_high = y_high_d

        for i in tqdm.tqdm(range(X.shape[1])):
            xi = X[:, i]

            if i in nominal_features:
                splitter = NominalSplitter(IMPROVEMENTS[self.criteria])
            else:
                splitter = NumericalSplitter(IMPROVEMENTS[self.criteria])

            improvement = splitter.find_best(
                xi, weighted_y_high)

            if improvement > self.res_['improvement']:
                self.res_['improvement'] = improvement
                self.res_['feature'] = i
                self.res_['splitter'] = splitter

        mask = self.res_['splitter'].split(X[:, self.res_['feature']])
        self.res_['classes'] = one.categories_[0]
        self.res_['left_prob'] = data_to_probs(y_high_d[mask])
        self.res_['right_prob'] = data_to_probs(y_high_d[~mask])
        return self

    def predict_proba(self, x):
        check_is_fitted(self, ['res_', ])
        mask = self.res_['splitter'].split(x[:, self.res_['feature']])
        return np.array([self.res_['left_prob'], self.res_['right_prob']])[np.where(mask, 0, 1)]

    def predict(self, x):
        prob = self.predict_proba(x)
        return self.res_['classes'][np.argmax(prob, axis=1)]

    def get_params(self, deep=True):
        return {"nominal_cols": self.nominal_cols,
                'criteria': self.criteria
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


__version__ = 'x'

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    import openml

    dataset_meta_info = openml.datasets.get_dataset(1457, False)
    nominal_cols = ','.join([str(key)
                             for key, value in dataset_meta_info.features.items() if value.data_type == 'nominal'])
    # clf = DecisionTreeClassifier(max_depth=1)
    clf = BsplitZClassifier(
        nominal_method='bsplitz',
        default_feature_type='nominal', non_default_feature_colmns=nominal_cols)
    task = openml.tasks.get_task(56571)
    run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)

    # The run may be stored offline, and the flow will be stored along with it:
    # run.to_filesystem(directory='new_myrun')

    # They may be loaded and uploaded at a later time
    # run = openml.runs.OpenMLRun.from_filesystem(directory='new_myrun')
    print(run)
    run.publish()
