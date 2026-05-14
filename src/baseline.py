from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score
)


def evaluate_baseline(X_train, X_test, y_train, y_test):

    # Majority class baseline
    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0)
    }

    print("\nBaseline Metrics")
    print(metrics)

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    return baseline, metrics