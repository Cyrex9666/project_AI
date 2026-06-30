from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from dataset import build_dataset
from model_eval_wkfwd import run_walk_forward_test, print_average_results, plot_average_metric, plot_walk_forward_metric

from model_eval_paramstb import (
    run_parameter_stability_test,
    print_parameter_stability_summary,
    plot_parameter_stability
)


# Run the walk-forward test
target_stock = "QQQ"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=4
)

models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1),
}

results_df = run_walk_forward_test(
    dataset=dataset,
    models=models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
)

print_average_results(results_df)
# # ===========================================================================
# plot_walk_forward_metric(results_df, metric="accuracy")
# plot_walk_forward_metric(results_df, metric="macro_f1")
# plot_average_metric(results_df, metric="macro_f1")

# # ---------------------------------------------------------
# # Parameter stability test
# # ---------------------------------------------------------

# coefficients_df = run_parameter_stability_test(
#     dataset=dataset,
#     model=LogisticRegression(max_iter=1000),
#     model_name="Logistic Regression",
#     train_window_years=5,
#     first_test_year=2015,
#     last_test_year=2025
# )

# print("\nCoefficient Values Over Time:")
# print(coefficients_df.round(4))

# print_parameter_stability_summary(coefficients_df)

# plot_parameter_stability(coefficients_df)