import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn import preprocessing


def prepare_data(three_class=[b'back.', b'nmap.', b'warezclient.'], col_in_experiment=2):
    x, y = fetch_kddcup99(return_X_y=True)
    rows = np.sum([y == i for i in three_class], axis=0)
    x_in = x[rows != 0, col_in_experiment]
    y_in = y[rows != 0]
    return x_in, y_in


def convert_vec(data):
    vec = preprocessing.LabelEncoder()
    return vec.fit_transform(data)


def gini_improvements(pis, q_weight):
  double_weights = q_weight**2
  q = (pis*q_weight).sum(axis=1)[:, np.newaxis]
  denorminator = q*(1-q)/double_weights
  return ((pis-q)**2/denorminator).sum(axis=1)


def main():
    d_x, d_y = prepare_data()
    x, y = convert_vec(d_x), convert_vec(d_y)
    labels, label_counts = np.unique(y, return_counts=True)
    y = preprocessing.OneHotEncoder().fit_transform(y[:, np.newaxis])
    categories, inverse,  count = np.unique(
        x, return_counts=True, return_inverse=True)
    y_sum = np.zeros((count.size, y.shape[1]))
    np.add.at(y_sum, inverse, y.toarray())
    pis = y_sum/label_counts
    q_weight = label_counts/label_counts.sum()
    print(gini_improvements(pis, q_weight))


if __name__ == "__main__":
    main()
