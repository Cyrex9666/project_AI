from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from dataset import build_dataset

from model_eval_wkfwd import (
    run_walk_forward_test_with_predictions,
    print_average_results,
    plot_walk_forward_confusion_matrices
)



# getting data
target_stock = "QQQ"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=4
)


# model comparison
models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1),
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
    "Gradient Boosting tuned": GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=25,
        subsample=0.75,
        max_features="sqrt",
        loss="log_loss",
        random_state=67
    )
}


# run walk forward test and store predictions
results_df, predictions_df = run_walk_forward_test_with_predictions(
    dataset=dataset,
    models=models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
)


print_average_results(results_df)

# print confusion matrices
plot_walk_forward_confusion_matrices(
    predictions_df,
    columns=2,
    model_order=[
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest tuned",
        "Gradient Boosting tuned"
    ]
)