import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from dataset import build_dataset
from model_prep import split_features_and_target

# Creates a fresh model for each walk-forward step.
#
# For normal sklearn models:
#     LogisticRegression()
#     RandomForestClassifier()
#     XGBClassifier()
# We use clone(model).
#
# For models that need to be rebuilt manually, like some neural networks,
# you can pass in a function/lambda that creates the model.
def create_fresh_model(model):

    # if model is a funciton and does not contain a "fit" attribute in its calling, then make a clone of the
    # object
    if callable(model) and not hasattr(model, "fit"):
        return model()

    return clone(model)




# Runs walk-forward testing.

# Example:
#     Train 2010-2014 -> Test 2015
#     Train 2011-2015 -> Test 2016
#     Train 2012-2016 -> Test 2017
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



# Plots one metric over each walk-forward test year.

# Good metrics to use:
# accuracy
# macro_precision
# macro_recall
# macro_f1
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


# Plots the average score for each model across all walk-forward steps
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


# Prints one clean summary table instead of printing every single year.
def print_average_results(results_df):
    average_results = (
        results_df
        .groupby("model")[["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
        .mean()
        .sort_values(by="macro_f1", ascending=False)
    )

    print("\nAverage Walk-Forward Results:")
    print(average_results.round(3))