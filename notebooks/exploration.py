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
    data_path = "./data/raw/loan_approval.csv"
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
    return x_train, x_val, y_train, y_val


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Evaluation

    * Fit model on full train set, evaluate on validation set
    * Cross Validation
    """)
    return


@app.cell
def _(x_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model_logistic_regression = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=1000, random_state=42
    )
    model_logistic_regression.fit(X=x_train, y=y_train)
    return LogisticRegression, model_logistic_regression


@app.cell
def _(model_logistic_regression, x_train, y_train):
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        classification_report,
    )

    def evaluate_model(model, x, y):
        y_pred = model_logistic_regression.predict(x)
        train_classification_report = classification_report(y, y_pred)
        print(f"Accuracy score: {accuracy_score(y, y_pred):.2f}")
        print(f"ROC AUC score: {roc_auc_score(y, y_pred):.2f}")
        print(f"Precision score: {precision_score(y, y_pred):.2f}")
        print(f"Recall score: {recall_score(y, y_pred):.2f}")
        print(train_classification_report)
        return train_classification_report

    evaluate_model(model_logistic_regression, x_train, y_train)
    return evaluate_model, roc_auc_score


@app.cell
def _(evaluate_model, model_logistic_regression, x_val, y_val):
    evaluate_model(model_logistic_regression, x_val, y_val)
    return


@app.cell
def _(LogisticRegression, roc_auc_score, x_train, x_val, y_train, y_val):
    from sklearn.model_selection import KFold
    import numpy as np

    full_train_X = np.concat([x_train.to_numpy(), x_val.to_numpy()])
    full_train_y = np.concat([y_train, y_val])

    def roc_scores_eval(full_train_X, full_train_y, C=1.0):
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        roc_scoreskf = []
        for i, (train_index, test_index) in enumerate(
            kf.split(full_train_X, full_train_y)
        ):
            x_train_kf, y_train_kf = (
                full_train_X[train_index],
                full_train_y[train_index],
            )
            x_val_kf, y_val_kf = full_train_X[test_index], full_train_y[test_index]
            model_kf = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
            model_kf.fit(x_train_kf, y_train_kf)
            y_val_proba_pred_kf = model_kf.predict_proba(x_val_kf)[:, 1]
            roc_scoreskf.append(roc_auc_score(y_val_kf, y_val_proba_pred_kf))
            del model_kf
        return np.std(roc_scoreskf), round(np.mean(roc_scoreskf), 3)

    roc_scores_eval(full_train_X, full_train_y)
    return full_train_X, full_train_y


@app.cell
def _(mo):
    mo.md(r"""
    ## Hyperparamater tuning
    """)
    return


@app.cell
def _(LogisticRegression, full_train_X, full_train_y):
    from sklearn.model_selection import GridSearchCV

    parameters = {"C": [0.01, 1, 10], "max_iter": [1000], "solver": ["liblinear"]}
    model_logistic_regression_gs = LogisticRegression()
    clf = GridSearchCV(model_logistic_regression_gs, parameters)
    clf.fit(full_train_X, full_train_y)
    return (clf,)


@app.cell
def _(clf, evaluate_model, x_train, y_train):
    evaluate_model(clf, x_train, y_train)
    return


@app.cell
def _(clf, evaluate_model, x_val, y_val):
    evaluate_model(clf, x_val, y_val)
    return


@app.cell
def _(clf):
    clf
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
