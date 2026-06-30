# this file will train and test a simple logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from dataset import build_dataset
from model_prep import prepare_model_data
from model_eval_helpers import evaluate_classifier, print_evaluation_results, plot_confusion_matrix, plot_learning_curve

# prep model data
target_stock = "CBA.AX"

dataset = build_dataset(
    ticker=target_stock,
    start_date="2010-01-01",
    end_date="2026-01-01",
    feature_set=1
)

X_train, X_test, y_train, y_test, scaler = prepare_model_data(dataset, train_size=0.8)

# naive baseline
model_majority = DummyClassifier(strategy="most_frequent")
model_majority.fit(X_train, y_train)

results_majority = evaluate_classifier(
    model=model_majority,
    X_test=X_test,
    y_test=y_test
)

# logistic regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

results_lr = evaluate_classifier(
    model=model_lr,
    X_test=X_test,
    y_test=y_test
)

#=================================================================================================================
# print results, single train/test split
#=================================================================================================================
print_evaluation_results(
    results_majority,
    model_name="Majority Class Baseline"
)

print_evaluation_results(
    results_lr,
    model_name="Logistic Regression"
)

plot_confusion_matrix(results_majority, model_name="Majority Class Baseline")
plot_confusion_matrix(results_lr, model_name="Logistic Regression")

plot_learning_curve(model_majority, X_train, y_train, X_test, y_test, model_name="Majority Class Baseline")
plot_learning_curve(model_lr, X_train, y_train, X_test, y_test, model_name="Logistic Regression")
