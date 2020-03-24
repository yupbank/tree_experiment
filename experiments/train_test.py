import warnings
import category_encoders
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_kddcup99
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss
from bsz.bsplitz import BsplitZClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import openml
import numpy as np
import time


def generate_candidates(adjusted_cols):
    return [
        (
            "bsplitz_method",
            Pipeline(
                [
                    ("cate", category_encoders.OrdinalEncoder(cols=adjusted_cols)),
                    (
                        "ordinal_encoder",
                        BsplitZClassifier(
                            adjusted_cols, random_state=10, num_samples=100
                        ),
                    ),
                ]
            ),
        ),
        (
            "target_encoder",
            Pipeline(
                [
                    (
                        "target_encoder",
                        category_encoders.TargetEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        #            ('clf', DecisionTreeClassifier(max_depth=1))])),
        (
            "m_encoder",
            Pipeline(
                [
                    (
                        "m_encoder",
                        category_encoders.MEstimateEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        #        ('clf', DecisionTreeClassifier(max_depth=1))])),
        (
            "cat_encoder",
            Pipeline(
                [
                    (
                        "m_encoder",
                        category_encoders.CatBoostEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        # ('clf', DecisionTreeClassifier(max_depth=1))])),
        # ('backward_encoder', Pipeline([
        #     ('backward_encoder', category_encoders.BackwardDifferenceEncoder(
        #         cols=adjusted_cols)),
        #     ('clf', BsplitZClassifier())])), #skip because of too slow
        (
            "basen_encoder",
            Pipeline(
                [
                    (
                        "basen_encoder",
                        category_encoders.BaseNEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        # ('clf', DecisionTreeClassifier(max_depth=1))])),
        (
            "binary_encoder",
            Pipeline(
                [
                    (
                        "basen_encoder",
                        category_encoders.BinaryEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        # ('clf', DecisionTreeClassifier(max_depth=1))])),
        (
            "count_encoder",
            Pipeline(
                [
                    (
                        "basen_encoder",
                        category_encoders.CountEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        # ('clf', DecisionTreeClassifier(max_depth=1))])),
        # ('hashing_encoder', Pipeline([
        #    ('basen_encoder', category_encoders.HashingEncoder(
        #        cols=adjusted_cols)),
        #    ('clf', BsplitZClassifier())])), #skip because of too slow
        # ('woe_encoder', Pipeline([
        #     ('woe_encoder', category_encoders.WOEEncoder(
        #         cols=adjusted_cols)),
        #     ('clf', BsplitZClassifier())])), #skip because of binary target only
        (
            "jamesstein_encoder",
            Pipeline(
                [
                    (
                        "js_encoder",
                        category_encoders.JamesSteinEncoder(cols=adjusted_cols),
                    ),
                    ("clf", BsplitZClassifier()),
                ]
            ),
        ),
        # ('clf', DecisionTreeClassifier(max_depth=1))])),
        # ('helmert_encoder', Pipeline([
        #    ('helmert_encoder', category_encoders.HelmertEncoder(
        #        cols=adjusted_cols)),
        #    ('clf', BsplitZClassifier())])), #skip because of too slow
    ]


def prepare_cate_encode2():
    import pandas as pd

    train_file = (
        "/Users/pengyu/Documents/tree_experiment/data/cate_challenge2/train.csv"
    )
    nominal_cols = range(10)
    names = ["nom_%s" % i for i in nominal_cols]
    df = pd.read_csv(train_file)
    x, y = df[names], df["target"]
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for n, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x.values[train_index]
        X_test = x.values[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        yield n, X_train, y_train, X_test, y_test, nominal_cols


def prepare_tami():
    dataset_info = openml.datasets.get_dataset(1505)
    d = dataset_info.get_data()[0]
    x = d.iloc[:, :-1]
    y = d.iloc[:, -1]
    nominal_cols = [
        key
        for key, value in dataset_info.features.items()
        if value.data_type == "nominal" and value.name != "Class"
    ]
    y = preprocessing.LabelEncoder().fit_transform(y)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for n, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x.values[train_index]
        X_test = x.values[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        yield n, X_train, y_train, X_test, y_test, nominal_cols


def prepare_kddcup():
    x, y = fetch_kddcup99(return_X_y=True)
    y = preprocessing.LabelEncoder().fit_transform(y)
    nominal_cols = [1, 2, 3]
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for n, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        X_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        yield n, X_train, y_train, X_test, y_test, nominal_cols


def prepare_cmc():
    data = fetch_openml("cmc", version=1)
    x, y = data["data"], data["target"]
    nominal_cols = []
    for i, feature in enumerate(data["feature_names"]):
        if feature in data["categories"]:
            nominal_cols.append(i)
    y = preprocessing.LabelEncoder().fit_transform(y)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for n, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        X_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        yield n, X_train, y_train, X_test, y_test, nominal_cols


def do_experiment(data_name, i, clfs, X_train, y_train, X_test, y_test):
    warnings.filterwarnings("ignore")
    for clf_name, clf in clfs:
        start = time.time()
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_logloss = str(log_loss(y_train, clf.predict_proba(X_train)))
        test_logloss = str(log_loss(y_test, clf.predict_proba(X_test)))
        train_accuracy = str(accuracy_score(y_train, y_train_pred))
        test_accuracy = str(accuracy_score(y_test, y_test_pred))
        train_n_rows, train_n_cols = map(str, X_train.shape)
        test_n_rows, test_n_cols = map(str, X_test.shape)
        n_class = str(np.unique(y_train).shape[0])
        # improvement = str(clf.steps[-1][-1].improvement_)
        duration_in_sec = str(time.time() - start)
        print(
            ",".join(
                [
                    data_name,
                    str(i),
                    train_n_rows,
                    train_n_cols,
                    test_n_rows,
                    test_n_cols,
                    n_class,
                    clf_name,
                    train_accuracy,
                    test_accuracy,
                    train_logloss,
                    test_logloss,
                    #        duration_in_sec
                ]
            )
        )


def main():
    datasets = [
        ("tami", prepare_tami()),
        ("kddcup", prepare_kddcup()),
        ("cmc", prepare_cmc()),
        ("cate_encode2", prepare_cate_encode2()),
    ]
    header = "data_set,fold,train_n_rows,train_n_cols,test_n_rows,test_n_cols,n_class,method,train_accuracy,test_accuracy,train_logloss,test_logloss"
    print(header)
    for name, dataset in datasets:
        for i, X_train, y_train, X_test, y_test, nominal_features in dataset:
            candidates = generate_candidates(range(len(nominal_features)))
            do_experiment(
                name,
                i,
                candidates,
                X_train[:, nominal_features],
                y_train,
                X_test[:, nominal_features],
                y_test,
            )


if __name__ == "__main__":
    main()
