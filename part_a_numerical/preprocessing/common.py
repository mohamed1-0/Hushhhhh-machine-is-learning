import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42
TARGET = "Price"


def load_data(path="data/car_price_prediction.csv"):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    df = df.drop(columns=["ID"], errors="ignore")

    df["Mileage"] = (df["Mileage"].astype(str)
                    .str.replace(" km", "", regex=False)
                    .astype(float))

    df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")

    df["Turbo"] = df["Engine volume"].astype(str).str.contains("Turbo").astype(int)
    df["Engine volume"] = (df["Engine volume"].astype(str)
                        .str.replace(" Turbo", "", regex=False)
                        .astype(float))

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes("number").columns
    cat_cols = X_train.select_dtypes(exclude="number").columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])