import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn import preprocessing
from cube_to_zonotope import zonotope_vertices, deduplicate


def prepare_data(three_class=[b'back.', b'nmap.', b'warezclient.'], col_in_experiment=2):
    x, y = fetch_kddcup99(return_X_y=True)
    rows = np.sum([y == i for i in three_class], axis=0)
    x_in = x[rows != 0, col_in_experiment]
    y_in = y[rows != 0]
    return x_in, y_in


def convert_vec(data):
    vec = preprocessing.LabelEncoder()
    return vec.fit_transform(data)


def convert_matrix(data):
    return np.apply_along_axis(convert_vec, 0, data)


def gini_improvements(pis, q_weight):
  double_weights = q_weight**2
  q = (pis*q_weight).sum(axis=1)[:, np.newaxis]
  denorminator = q*(1-q)/double_weights
  return ((pis-q)**2/denorminator).sum(axis=1)


def main():
    d_x, d_y = fetch_kddcup99(return_X_y=True)
    x, y = convert_matrix(d_x), convert_vec(d_y)
    labels, label_counts = np.unique(y, return_counts=True)
    q_weight = label_counts/label_counts.sum()
    y = preprocessing.OneHotEncoder(
        sparse=False).fit_transform(y[:, np.newaxis])
    best_one_hot, best_search = None, None
    best_one_hot_improve, best_search_improve = 0, 0
    for i in range(x.shape[1]):
        if i in [19, 20]:
            continue
        x_i = x[:, i]
        categories, inverse,  count = np.unique(
            x_i, return_counts=True, return_inverse=True)
        y_sum = np.zeros((count.size, y.shape[1]))
        np.add.at(y_sum, inverse, y)
        pis = y_sum/label_counts
        improvements = gini_improvements(pis, q_weight)
        best = np.max(improvements)
        best_one_hot_improve = max(best_one_hot_improve, best)
        if best_one_hot_improve == best:
            best_one_hot = i
        print(i, best)

        piis = zonotope_vertices(deduplicate(pis), 5000, 1)
        improvements = gini_improvements(piis, q_weight)
        print('better', i, np.max(improvements[1:-1]))
        best = np.max(improvements)
        best_search_improve = max(best_search_improve, best)
        if best_search_improve == best:
            best_search = i
    print (best_one_hot, best_one_hot_improve)
    print (best_search, best_search_improve)


if __name__ == "__main__":
    main()
