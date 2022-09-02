import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from openml import extensions
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

class BszTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        print(X, '!!')
        self.d_ = y.mean(), y.median()
        return self

    def transform(self, X, y=None):
        val = np.zeros((X.shape[0], 2))
        val[:,0], val[:, 1] = self.d_
        X['1'] = val[:,0]
        X['2'] = val[:,1]
        return X


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 2], 'c': [1, 2, 3], 'd': ['x', 's', 's']})
    pipe = make_pipeline(make_column_transformer((BszTransformer(), extensions.sklearn.cat)))
    pipe.fit_transform(df, df['a'])
    d = pipe.transform(df)
    print(d)
