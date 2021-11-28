from pathlib import Path
from typing import List, Tuple

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from wine_platform.data import get_raw_path


def load_train_data():
    train_csv_path = Path(get_raw_path()).joinpath("winequalityN.csv")
    return pd.read_csv(train_csv_path)


def wine_na_columns() -> List[str]:
    """Returns the list of columns with missing values"""
    return [
        "fixed acidity",
        "pH",
        "volatile acidity",
        "sulphates",
        "citric acid",
        "residual sugar",
        "chlorides",
    ]


def clean_data(df: pd.DataFrame):
    """Fills na values with the mean (inplace operation)"""
    for col in wine_na_columns():
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)


def prepare_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.get_dummies(df, drop_first=True)
    df = df.rename(columns={"type_white": "wine_type"})
    # Transforming into a binary problem
    df["wine_quality"] = [1 if x > 6 else 0 for x in df.quality]

    y = df["wine_quality"]
    x = df.drop(["quality", "wine_quality"], axis=1)
    return x, y


def train_logistic_regression(
    X_train, y_train, X_test, y_test
) -> sklearn.base.ClassifierMixin:
    clf = LogisticRegression(solver="liblinear")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    clf.score(X_train, y_train)

    return clf, accuracy_score(y_test, y_pred)


def train():
    train_df = load_train_data()
    clean_data(train_df)
    X, y = prepare_for_training(train_df)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    clf, log_acc = train_logistic_regression(X_train, X_test, y_train, y_test)
    print(log_acc)


if __name__ == "__main__":
    train()
