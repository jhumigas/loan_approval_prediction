# Todo: trains model and saves artifact
import pickle
from loguru import logger
import sklearn
from loan_approval_prediction.config import ProjectConfig
import polars as pl


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


def clean_dataset(X: pl.DataFrame, y: pl.DataFrame, num_cols: list[str]):
    return X.select(num_cols), y.cast(int).to_numpy()


def evaluate_model(model, X_val: pl.DataFrame, y_val: pl.DataFrame):
    y_pred = model.predict(X_val)
    model_classification_report = sklearn.metrics.classification_report(y_val, y_pred)
    logger.info("Model evaluation report:\n" + model_classification_report)
    return model_classification_report


def save_model(model, model_path: str):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at {model_path}")
