import pickle
from loguru import logger
import sklearn
import polars as pl
import numpy as np


def load_data(data_path: str):
    """Load dataset from the specified path."""
    logger.info(f"Loading data from {data_path}")
    data = pl.read_csv(data_path)
    logger.info("Data loaded successfully")
    return data


def split_label(dataset: pl.dataframe, target_col: str):
    return dataset.drop(target_col), dataset[target_col]


def split_dataset(dataset: pl.dataframe, val_size=0.2, test_size=0.2, seed=1):
    train_size = 1 - test_size
    full_train_dataset, test_dataset = sklearn.model_selection.train_test_split(
        dataset, test_size=test_size, random_state=seed
    )
    train_dataset, val_dataset = sklearn.model_selection.train_test_split(
        full_train_dataset, test_size=val_size / train_size, random_state=seed
    )
    return train_dataset, val_dataset, test_dataset


def concatenate_datasets(
    X1: pl.DataFrame, y1: np.ndarray, X2: pl.DataFrame, y2: np.ndarray
):
    X_concat = pl.concat([X1, X2])
    y_concat = np.concat([y1, y2])
    return X_concat, y_concat


def clean_dataset(X: pl.DataFrame, y: pl.DataFrame, num_cols: list[str]):
    return X.select(num_cols), y.cast(int).to_numpy()


def create_model_instance(C: float, max_iter: int, solver: str, random_state: int):
    model = sklearn.linear_model.LogisticRegression(
        C=C, max_iter=max_iter, solver=solver, random_state=random_state
    )
    return model


def train_model(model, x_train: pl.DataFrame, y_train: pl.DataFrame):
    model.fit(X=x_train, y=y_train)
    return model


def evaluate_model(model, X_val: pl.DataFrame, y_val: pl.DataFrame):
    y_pred = model.predict(X_val)
    model_classification_report = sklearn.metrics.classification_report(y_val, y_pred)
    roc_auc = sklearn.metrics.roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    logger.info("Model evaluation report:\n" + model_classification_report)
    return roc_auc


def save_model(model, model_path: str):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at {model_path}")
    return model_path
