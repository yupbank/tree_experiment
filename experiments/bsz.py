import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pandas.core.dtypes import dtypes
from openml import extensions
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn import preprocessing

def bool_sign(a):

def aggregate_generators_by_direction(x):
    normal_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    unique_vector, index, inverse = np.unique(
        normal_x, axis=0, return_index=True, return_inverse=True
    )
    res = np.zeros_like(unique_vector)
    np.add.at(res, inverse, x)
    return res, inverse

def sample_zonotope_vertices(generators, n_samples=50, n_batches=1):
    m, n = generators.shape
    res = np.empty([0, m])
    for i in range(n_batches):
        samples = np.random.normal(size=(n_samples, n))
        S = np.sign(samples.dot(generators.T))
        res = np.unique(np.vstack((res, S, -S)), axis=0)
    return np.where(np.sign(res) > 0, 1.0, 0.0)

def convert_feature(vec, selection, xi):
    return vec.transform(xi).dot(selection.T)

def agg_feature_converter(Xi, y_vec):
    vec = preprocessing.OneHotEncoder(handle_unknown="ignore")
    xi = vec.fit_transform(Xi.values.to_numpy()[:,np.newaxis])
    raw_gs =  xi.T.dot(y_vec)
    #gs, inv = aggregate_generators_by_direction(raw_gs)
    selections = sample_zonotope_vertices(raw_gs)
    selected_items = selections.sum(axis=1)
    selections = selections[np.logical_and(selected_items>0, selected_items<raw_gs.shape[0])]
    return vec, selections

class BszTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.n_features = []
        self.categories_ = []
        self.converters_ = []
        label_encoder = preprocessing.OneHotEncoder(sparse=False)
        y_vec = label_encoder.fit_transform(y.values.to_numpy()[:,np.newaxis])
        for i, cate in enumerate(X.dtypes):
            if isinstance(cate, dtypes.CategoricalDtype):
                self.n_features.append(i)
                self.categories_.append(cate.categories)
        for n, cates in zip(self.n_features, self.categories_):
            feat = X.iloc[:, n]
            self.converters_.append(agg_feature_converter(feat))
        return self

    def transform(self, X, y=None):
        new_cols = []
        for n, cates, converts in zip(self.n_features, self.categories_, self.converters_):
            feat = X.iloc[:, n]
            new_cols.append(convert_feature(*converts, feat))
        return np.hstack(new_cols)


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 2], 'c': [1, 2, 3], 'd': ['x', 's', 's']})
    pipe = make_pipeline(make_column_transformer((BszTransformer(), extensions.sklearn.cat)))
    pipe.fit_transform(df, df['a'])
    d = pipe.transform(df)
    print(d)
