import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from dataset import build_dataset
from model_prep import split_features_and_target, time_train_test_split

# converts random forest feature importances into a clean table
def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df["importance_percent"] = (
        importance_df["importance"] / importance_df["importance"].sum()
    ) * 100

    return importance_df

# plots feature importance as a horizontal bar chart
def plot_feature_importance(importance_df, model_name="Random Forest"):
    plot_df = importance_df.sort_values(
        by="importance",
        ascending=True
    )

    plt.barh(plot_df["feature"], plot_df["importance"])

    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x")
    plt.show()

# dataset
target_stock = "QQQ"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=4
)


# split n train n test
X, y = split_features_and_target(dataset)

X_train, X_test, y_train, y_test = time_train_test_split(
    X=X,
    y=y,
    train_size=0.8
)

# trainingg
model_rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=100,
    max_features="sqrt",
    class_weight="balanced",
    random_state=67,
    n_jobs=-1
)

model_rf.fit(X_train, y_train)

importance_df = get_feature_importance(
    model=model_rf,
    feature_names=X.columns
)

print("\nRandom Forest Feature Importance:")
print(importance_df.round(4))

plot_feature_importance(
    importance_df,
    model_name="Random Forest Tuned"
)