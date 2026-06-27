import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.base import clone
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
def plot_confusion_matrix(results, model_name="Model"):
    """
    Plots the confusion matrix using matplotlib.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=["Down/Hold", "Up"]
    )

    display.plot()

    plt.title(f"{model_name} Confusion Matrix")
    plt.show()


def plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains the model on increasing amounts of training data
    and plots test accuracy and macro F1-score.

    This is useful for seeing whether the model improves
    as it receives more historical training data.
    """

    train_sizes = np.linspace(0.1, 1.0, 20)

    accuracy_scores = []
    macro_f1_scores = []
    sample_sizes = []

    for train_size in train_sizes:
        subset_size = int(len(X_train) * train_size)

        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train.iloc[:subset_size]

        model_copy = clone(model)
        model_copy.fit(X_train_subset, y_train_subset)

        y_pred = model_copy.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        accuracy_scores.append(accuracy)
        macro_f1_scores.append(macro_f1)
        sample_sizes.append(subset_size)

    plt.plot(sample_sizes, accuracy_scores, marker="o", label="Accuracy")
    plt.plot(sample_sizes, macro_f1_scores, marker="o", label="Macro F1-score")

    plt.title(f"{model_name} Learning Curve")
    plt.xlabel("Number of training samples")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluates a trained binary classification model.

    Class meanings:
        0 = price does not go up / negative class
        1 = price goes up / positive class

    Returns:
        Dictionary containing the model's predictions and metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),

        # Macro averages treat classes 0 and 1 equally.
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
        ),

        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Down/Hold", "Up"],
            zero_division=0
        ),

        "predictions": y_pred
    }

    return metrics

def print_evaluation_results(results, model_name="Model"):
    """
    Prints all evaluation results in a consistent format.
    """

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'=' * 50}")

    print(f"Accuracy:        {results['accuracy']:.3f}")
    print(f"Macro precision: {results['macro_precision']:.3f}")
    print(f"Macro recall:    {results['macro_recall']:.3f}")
    print(f"Macro F1-score:  {results['macro_f1']:.3f}")

    print("\nConfusion matrix:")
    print(results["confusion_matrix"])

    print("\nClassification report:")
    print(results["classification_report"])