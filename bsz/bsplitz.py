import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from bsz.utils import split_data, bsplitz_method, numerical_method, gini_improvements, data_to_generators
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import tqdm


def split_nominal(xi, weighted_y_high, d):
    vectorizer = OneHotEncoder()
    xi_high_d = vectorizer.fit_transform(xi[:, np.newaxis])
    gs_not_normalized = xi_high_d.T.dot(weighted_y_high).A
    gs = gs_not_normalized/d
    pis, indices = bsplitz_method(gs)
    improvements = np.nan_to_num(gini_improvements(pis, d))
    best_index = np.argmax(improvements)
    improvement = improvements[best_index]
    feature_threshold = indices[best_index]
    return improvement, feature_threshold, vectorizer


def split_numerical(xi, weighted_y_high, d):
    orders = np.argsort(xi)
    pis = np.cumsum(weighted_y_high[orders], axis=1)
    improvements = np.nan_to_num(gini_improvements(pis, d))
    best_index = np.argmax(improvements)
    improvement = improvements[best_index]
    feature_threshold = indices[best_index]
    return improvement, xi[orders[best_index]], None


class BsplitZClassifier(BaseEstimator, ClassifierMixin):
    """
    BSplitZ Decision Stump classifier supports native nominal features
    """

    def __init__(self, nominal_cols='', criteria='gini'):
        """

        :param nominal_cols: comma separated columns of nominal features, if not specified treat evey feature as nomial
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
        y_high_d = OneHotEncoder().fit_transform(y[:, np.newaxis])
        if sample_weight is not None:
            weighted_y_high = y_high_d.multiply(sample_weight[:, np.newaxis])
        else:
            weighted_y_high = y_high_d

        d = weighted_y_high.sum(axis=0).A.ravel()
        for i in tqdm.tqdm(range(X.shape[1])):
            xi = X[:, i]

            if i in nominal_features:
                improvement, threshold, vectorizer = split_nominal(
                    xi, weighted_y_high, d)
            else:
                improvement, threshold = split_numerical(
                    xi, weighted_y_high, d)

            if improvement > self.res_['improvement']:
                self.res_['improvement'] = improvement
                self.res_['feature'] = i
                self.res_['feature_threshold'] = feature_threshold
                self.res_['vectorizer'] = vectorizer

        if self.res_['feature'] in nominal_features:
            enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            self.res_['vectorizer'] = enc.fit(
                X[:, self.res_['feature']][:, np.newaxis])
        else:
            return
        mask = split_data(X[:, self.res_['feature']],
                          self.res_['feature_threshold'],
                          self.res_['vectorizer'],
                          self.res_['type'])
        left_classes = y_high_d[mask]
        left_probs = np.nan_to_num(left_classes.sum(0)/np.sum(left_classes))

        right_classes = y_high_d[~mask]
        right_probs = np.nan_to_num(right_classes.sum(0)/np.sum(right_classes))
        self.classes_ = labels
        self.res_['classes'] = labels
        self.res_['left_prob'] = left_probs
        self.res_['right_prob'] = right_probs
        return self

    def predict_proba(self, x):
        check_is_fitted(self, ['res_', ])
        mask = split_data(x[:, self.res_['feature']],
                          self.res_['feature_threshold'], self.res_['vectorizer'], self.res_['type'])
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
    #clf = DecisionTreeClassifier(max_depth=1)
    clf = BsplitZClassifier(
        nominal_method='bsplitz',
        default_feature_type='nominal', non_default_feature_colmns=nominal_cols)
    task = openml.tasks.get_task(56571)
    run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)

# The run may be stored offline, and the flow will be stored along with it:
    #run.to_filesystem(directory='new_myrun')

# They may be loaded and uploaded at a later time
    #run = openml.runs.OpenMLRun.from_filesystem(directory='new_myrun')
    print(run)
    run.publish()
