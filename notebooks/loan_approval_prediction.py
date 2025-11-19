import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import sklearn
    import seaborn as sns
    from sklearn.metrics import mutual_info_score

    return mo, pl, sklearn, sns


@app.cell
def _(mo):
    mo.md(r"""
    ## Data collection
    """)
    return


@app.cell
def _(pl):
    data_path = "./data/dataset/loan_approval.csv"
    raw_data = pl.read_csv(data_path)
    raw_data
    return (raw_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    cat_cols = ["city", "name"]
    num_cols = ["income", "credit_score", "loan_amount", "years_employed", "points"]
    target_col = "loan_approved"
    return num_cols, target_col


@app.cell
def _(raw_data):
    raw_data.describe()
    return


@app.cell
def _(pl, raw_data, sns, target_col):
    # Are the labels imbalanced ?
    sns.histplot(data=raw_data.select(pl.col(target_col)))
    return


@app.cell
def _(pl, raw_data, sns):
    # Credit score distribution
    sns.histplot(data=raw_data.select(pl.col("credit_score")))
    return


@app.cell
def _(raw_data):
    # Missing value count
    raw_data.null_count()
    return


@app.cell
def _(pl, raw_data, target_col):
    # Feature importance using Correlation between feature and target variables
    raw_data.select(
        pl.struct(pl.corr(pl.all(), pl.col(target_col))).alias(
            f"corr_with_{target_col}"
        )
    ).unpivot().unnest("value")
    return


@app.cell
def _(pl, raw_data):
    # Check what the feature engineering would entail
    raw_data.select(
        pl.corr(a=pl.col("loan_amount") / pl.col("income"), b="loan_approved")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Validation framework setup
    """)
    return


@app.cell
def _(pl, raw_data, target_col):
    def split_label(dataset: pl.dataframe, target_col: str):
        return dataset.drop(target_col), dataset[target_col]

    full_x, full_y = split_label(raw_data, target_col=target_col)
    return full_x, full_y


@app.cell
def _(full_x, full_y, pl, sklearn):
    def split_dataset(dataset: pl.dataframe, val_size=0.2, test_size=0.2, seed=1):
        train_size = 1 - test_size
        full_train_dataset, test_dataset = sklearn.model_selection.train_test_split(
            dataset, test_size=test_size, random_state=seed
        )
        train_dataset, val_dataset = sklearn.model_selection.train_test_split(
            full_train_dataset, test_size=val_size / train_size, random_state=seed
        )
        return train_dataset, val_dataset, test_dataset

    x_raw_train, x_raw_val, x_raw_test = split_dataset(
        full_x, val_size=0.2, test_size=0.2, seed=1
    )
    y_raw_train, y_raw_val, y_raw_test = split_dataset(
        full_y, val_size=0.2, test_size=0.2, seed=1
    )
    return (
        x_raw_test,
        x_raw_train,
        x_raw_val,
        y_raw_test,
        y_raw_train,
        y_raw_val,
    )


@app.cell
def _(
    num_cols,
    pl,
    x_raw_test,
    x_raw_train,
    x_raw_val,
    y_raw_test,
    y_raw_train,
    y_raw_val,
):
    def clean_dataset(X: pl.DataFrame, y: pl.DataFrame, num_cols: list[str]):
        return X.select(num_cols), y.cast(int).to_numpy()

    x_train, y_train = clean_dataset(x_raw_train, y_raw_train, num_cols)
    x_val, y_val = clean_dataset(x_raw_val, y_raw_val, num_cols)
    x_test, y_test = clean_dataset(x_raw_test, y_raw_test, num_cols)
    return x_train, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Evaluation
    """)
    return


@app.cell
def _(x_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model_logistic_regression = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=1000, random_state=42
    )
    model_logistic_regression.fit(X=x_train, y=y_train)
    return (model_logistic_regression,)


@app.cell
def _(model_logistic_regression, x_train, y_train):
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        classification_report,
    )

    y_pred_train = model_logistic_regression.predict(x_train)
    train_classification_report = classification_report(y_train, y_pred_train)
    print(f"Accuracy score on train set: {accuracy_score(y_train, y_pred_train):.2f}")
    print(f"ROC AUC score on train set: {roc_auc_score(y_train, y_pred_train):.2f}")
    print(f"Precision score on train set: {precision_score(y_train, y_pred_train):.2f}")
    print(f"Recall score on train set: {recall_score(y_train, y_pred_train):.2f}")
    print(train_classification_report)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving model
    """)
    return


@app.cell
def _(model_logistic_regression):
    import pickle

    model_folder = "./models"
    model_name = "model_logistic_regression_C=1.0.bin"
    with open(f"{model_folder}/{model_name}", "wb") as f_out:
        pickle.dump(model_logistic_regression, f_out)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
