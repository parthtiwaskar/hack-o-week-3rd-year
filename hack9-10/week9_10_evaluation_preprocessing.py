"""Week 9–10: evaluate classifiers and prepare imperfect data.

Run with: python week9_10_evaluation_preprocessing.py
Requires numpy and scikit-learn.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def add_engineered_features(features: np.ndarray) -> np.ndarray:
    """Append a feature interaction while preserving the original columns."""
    interaction = (features[:, 0] * features[:, 1]).reshape(-1, 1)
    return np.column_stack((features, interaction))


def make_incomplete_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible binary dataset with missing numeric values."""
    features, target = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.1,
        random_state=29,
    )
    features = add_engineered_features(features)

    random = np.random.default_rng(29)
    missing_mask = random.random(features.shape) < 0.08
    features[missing_mask] = np.nan
    return features, target


def build_model_pipeline() -> object:
    """Build preprocessing and classification as one leakage-safe estimator."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=29),
    )


def evaluate_holdout(
    model: object,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    """Fit on training data and report classification metrics on held-out data."""
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, 1]
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    scores = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision": float(precision_score(test_y, predictions, zero_division=0)),
        "recall": float(recall_score(test_y, predictions, zero_division=0)),
        "f1": float(f1_score(test_y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
    }

    print("Train/test evaluation (positive class = 1)")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    for metric, score in scores.items():
        print(f"{metric:10} {score:.3f}")
    return scores


def cross_validation_scores(features: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Score the complete pipeline across stratified folds."""
    model = build_model_pipeline()
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(
        model,
        features,
        target,
        cv=5,
        scoring=scoring,
        return_train_score=False,
    )
    means = {
        metric: float(np.mean(results[f"test_{metric}"]))
        for metric in scoring
    }

    print("\n5-fold cross-validation (mean score across folds)")
    for metric, score in means.items():
        print(f"{metric:10} {score:.3f}")
    return means


def main() -> None:
    features, target = make_incomplete_dataset()
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=29,
        stratify=target,
    )

    model = build_model_pipeline()
    holdout_scores = evaluate_holdout(model, train_x, test_x, train_y, test_y)
    cv_scores = cross_validation_scores(features, target)

    assert np.isnan(features).any()
    assert all(0.0 <= score <= 1.0 for score in holdout_scores.values())
    assert all(0.0 <= score <= 1.0 for score in cv_scores.values())
    print("\nPipeline handled missing values; all metric checks passed.")


if __name__ == "__main__":
    main()