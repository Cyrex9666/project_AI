import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from model_prep import split_features_and_target

# Runs a parameter stability test across walk-forward training windows.
# Only used for models with coeffs
def run_parameter_stability_test(
    dataset,
    model,
    model_name="Model",
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
):
    X, y = split_features_and_target(dataset)

    coefficient_rows = []

    # lowk copying the wkfwd code structure from before: for year, except we dont have to test
    for test_year in range(first_test_year, last_test_year + 1):
        train_start_year = test_year - train_window_years
        train_end_year = test_year

        # just training no testing, training is sufficient enough to find coeffs
        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-01-01"

        train_mask = (X.index >= train_start) & (X.index < train_end)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        if len(X_train) == 0:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # trainingg
        model_copy = clone(model)
        model_copy.fit(X_train_scaled, y_train)

        if not hasattr(model_copy, "coef_"):
            raise ValueError(
                f"{model_name} does not have coefficients."
                "Parameter stability can only work for models like Logistic Regression."
            )

        row = {
            "model": model_name,
            "test_year": test_year,
            "train_start_year": train_start_year,
            "train_end_year": train_end_year - 1,
            "intercept": model_copy.intercept_[0]
        }

        coefficients = model_copy.coef_[0]

        for feature_name, coefficient in zip(X.columns, coefficients):
            row[feature_name] = coefficient

        coefficient_rows.append(row)

    coefficients_df = pd.DataFrame(coefficient_rows)

    return coefficients_df

# Summarises how stable each feature coefficient is across time
def summarize_parameter_stability(coefficients_df):
    if coefficients_df.empty:
        print("No coefficient data found.")
        return pd.DataFrame()

    non_feature_columns = {
        "model",
        "test_year",
        "train_start_year",
        "train_end_year",
        "intercept"
    }

    feature_columns = [
        column for column in coefficients_df.columns
        if column not in non_feature_columns
    ]

    summary_rows = []

    for feature in feature_columns:
        values = coefficients_df[feature]

        positive_years = (values > 0).sum()
        negative_years = (values < 0).sum()
        zero_years = (values == 0).sum()
        total_years = len(values)

        sign_consistency = max(positive_years, negative_years) / total_years

        mean_coefficient = values.mean()
        std_coefficient = values.std()

        if std_coefficient == 0:
            stability_ratio = float("inf")
        else:
            stability_ratio = abs(mean_coefficient) / std_coefficient

        summary_rows.append({
            "feature": feature,
            "mean_coefficient": mean_coefficient,
            "std_coefficient": std_coefficient,
            "positive_years": positive_years,
            "negative_years": negative_years,
            "zero_years": zero_years,
            "sign_consistency": sign_consistency,
            "stability_ratio": stability_ratio
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=["sign_consistency", "stability_ratio"],
        ascending=False
    )

    return summary_df

# Prints a clean parameter stability summary table
def print_parameter_stability_summary(coefficients_df):
    summary_df = summarize_parameter_stability(coefficients_df)

    if summary_df.empty:
        return

    print("\nParameter Stability Summary:")
    print(summary_df.round(4))

# Plots coefficient values over walk-forward test years
def plot_parameter_stability(coefficients_df, features=None):
    if coefficients_df.empty:
        print("No coefficients to plot.")
        return

    non_feature_columns = {
        "model",
        "test_year",
        "train_start_year",
        "train_end_year",
        "intercept"
    }

    feature_columns = [
        column for column in coefficients_df.columns
        if column not in non_feature_columns
    ]

    if features is None:
        features = feature_columns

    for feature in features:
        plt.plot(
            coefficients_df["test_year"],
            coefficients_df[feature],
            marker="o",
            label=feature
        )

    plt.axhline(0, linestyle="--")

    plt.title("Parameter Stability Over Time")
    plt.xlabel("Test year")
    plt.ylabel("Logistic Regression Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()