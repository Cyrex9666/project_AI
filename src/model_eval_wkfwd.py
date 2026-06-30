import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from dataset import build_dataset
from model_prep import split_features_and_target

# creates a fresh model for each walk-forward step
def create_fresh_model(model):

    # if model is a funciton and does not contain a "fit" attribute in its calling, then make a clone of the
    # object
    if callable(model) and not hasattr(model, "fit"):
        return model()

    return clone(model)

# runs walk forward testing
def run_walk_forward_test(
    dataset,
    models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
):

    X, y = split_features_and_target(dataset)

    all_results = []

    # for each year
    for test_year in range(first_test_year, last_test_year + 1):
        train_start_year = test_year - train_window_years
        train_end_year = test_year

        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-01-01"

        test_start = f"{test_year}-01-01"
        test_end = f"{test_year + 1}-01-01"

        # only select rows from either the train or test section
        train_mask = (X.index >= train_start) & (X.index < train_end)
        test_mask = (X.index >= test_start) & (X.index < test_end)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # scalerr
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # for each model, 
        for model_name, model in models.items():
            model_copy = create_fresh_model(model)

            model_copy.fit(X_train_scaled, y_train)
            y_pred = model_copy.predict(X_test_scaled)

            all_results.append({
                "model": model_name,
                "test_year": test_year,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year - 1,
                "train_rows": len(X_train),
                "test_rows": len(X_test),

                "accuracy": accuracy_score(y_test, y_pred),
                "macro_precision": precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                "macro_recall": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                "macro_f1": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            })

    results_df = pd.DataFrame(all_results)

    return results_df



# plots one metric over each walk-forward test year
def plot_walk_forward_metric(results_df, metric):

    for model_name in results_df["model"].unique():
        model_results = results_df[results_df["model"] == model_name]

        plt.plot(
            model_results["test_year"],
            model_results[metric],
            marker="o",
            label=model_name
        )

    plt.title(f"Walk-Forward {metric}")
    plt.xlabel("Test year")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()


# plots the average score for each model across all walk forward steps
def plot_average_metric(results_df, metric):
    average_results = (
        results_df
        .groupby("model")[metric]
        .mean()
        .sort_values(ascending=False)
    )

    average_results.plot(kind="bar")

    plt.title(f"Average Walk-Forward {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()


# prints one clean summary table instead of printing every single year
def print_average_results(results_df):
    average_results = (
        results_df
        .groupby("model")[["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
        .mean()
        .sort_values(by="macro_f1", ascending=False)
    )

    print("\nAverage Walk-Forward Results:")
    print(average_results.round(3))

# runs walk forward testing and stores both aggregate results
# and every prediction made by each model, this is for confusion matrices
def run_walk_forward_test_with_predictions(
    dataset,
    models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
):
    X, y = split_features_and_target(dataset)

    all_results = []
    all_predictions = []

    for test_year in range(first_test_year, last_test_year + 1):
        train_start_year = test_year - train_window_years
        train_end_year = test_year

        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-01-01"

        test_start = f"{test_year}-01-01"
        test_end = f"{test_year + 1}-01-01"

        train_mask = (X.index >= train_start) & (X.index < train_end)
        test_mask = (X.index >= test_start) & (X.index < test_end)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in models.items():
            model_copy = create_fresh_model(model)

            model_copy.fit(X_train_scaled, y_train)
            y_pred = model_copy.predict(X_test_scaled)

            all_results.append({
                "model": model_name,
                "test_year": test_year,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year - 1,
                "train_rows": len(X_train),
                "test_rows": len(X_test),

                "accuracy": accuracy_score(y_test, y_pred),
                "macro_precision": precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                "macro_recall": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                "macro_f1": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            })

            for date, actual_value, predicted_value in zip(
                X_test.index,
                y_test.values,
                y_pred
            ):
                all_predictions.append({
                    "model": model_name,
                    "test_year": test_year,
                    "date": date,
                    "actual": int(actual_value),
                    "predicted": int(predicted_value)
                })

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)

    return results_df, predictions_df

# prints one aggregated walk forward confusion matrix per model
def print_walk_forward_confusion_matrices(predictions_df):
    for model_name in predictions_df["model"].unique():
        model_predictions = predictions_df[
            predictions_df["model"] == model_name
        ]

        matrix = confusion_matrix(
            model_predictions["actual"],
            model_predictions["predicted"],
            labels=[0, 1]
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Down/Hold", "Actual Up"],
            columns=["Predicted Down/Hold", "Predicted Up"]
        )

        print(f"\n{model_name} Aggregated Walk-Forward Confusion Matrix:")
        print(matrix_df)

# plots aggregated walk forward confusion matrices in one figure
def plot_walk_forward_confusion_matrices(
        predictions_df,
        columns=2,
        model_order=None,
        normalize=False
    ):

        if model_order is None:
            model_names = list(predictions_df["model"].unique())
        else:
            model_names = [
                model_name for model_name in model_order
                if model_name in predictions_df["model"].unique()
            ]

        number_of_models = len(model_names)
        rows = (number_of_models + columns - 1) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 4 * rows)
        )

        # mkes axes easy to loop over whether we have 1 row or many rows
        if number_of_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, model_name in zip(axes, model_names):
            model_predictions = predictions_df[
                predictions_df["model"] == model_name
            ]

            if normalize:
                matrix = confusion_matrix(
                    model_predictions["actual"],
                    model_predictions["predicted"],
                    labels=[0, 1],
                    normalize="true"
                )
                value_format = ".2f"
            else:
                matrix = confusion_matrix(
                    model_predictions["actual"],
                    model_predictions["predicted"],
                    labels=[0, 1]
                )
                value_format = "d"

            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Down/Hold", "Up"]
            )

            display.plot(
                ax=ax,
                colorbar=False,
                values_format=value_format
            )

            ax.set_title(model_name)

        # hide empty subplot spaces if number of models is not perfectly divisible
        for unused_ax in axes[number_of_models:]:
            unused_ax.axis("off")

        if normalize:
            title = "Aggregated Walk-Forward Confusion Matrices - Normalized"
        else:
            title = "Aggregated Walk-Forward Confusion Matrices"

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()