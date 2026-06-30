from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from dataset import build_dataset

from model_eval_wkfwd import (
    run_walk_forward_test_with_predictions,
    print_average_results,
    print_walk_forward_confusion_matrices,
    plot_walk_forward_confusion_matrices
)


# build dataset
target_stock = "QQQ"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=4
)

# models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest initial": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest tuned": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        min_samples_split=100,
        max_features=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# run walk foerward n store predictions
results_df, predictions_df = run_walk_forward_test_with_predictions(
    dataset=dataset,
    models=models,
    train_window_years=5,
    first_test_year=2015,
    last_test_year=2025
)

# printing results
print("\nFull Walk-Forward Results:")
print(results_df.round(3))

print_average_results(results_df)

# confusion matrix
print_walk_forward_confusion_matrices(predictions_df)

plot_walk_forward_confusion_matrices(predictions_df)