from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from dataset import build_dataset
from model_eval_wkfwd import (
    run_walk_forward_test,
    print_average_results,
    plot_walk_forward_metric,
    plot_average_metric
)

# dataset
target_stock = "QQQ"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=4
)

# choosing models / fine tuning
models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest tuned": RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=67,
        n_jobs=-1
    ),

        "Random Forest initial": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        random_state=67,
        n_jobs=-1
    )
}

# walkfwd comparison
results_df = run_walk_forward_test(
    dataset=dataset,
    models=models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
)

# print n plot

print("\nFull Walk-Forward Results:")
print(results_df.round(3))

print_average_results(results_df)

plot_walk_forward_metric(results_df, metric="accuracy")
plot_walk_forward_metric(results_df, metric="macro_f1")
plot_average_metric(results_df, metric="macro_f1")