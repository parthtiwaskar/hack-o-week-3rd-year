"""Week 7–8: regression and classification with scikit-learn.

Run with: python week7_8_supervised_learning.py
Requires numpy and scikit-learn.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def regression_examples() -> dict[str, tuple[float, float]]:
    """Compare linear, polynomial, Ridge, and Lasso regression."""
    features, target = make_regression(
        n_samples=240,
        n_features=1,
        noise=18.0,
        random_state=17,
    )
    # Add a curved signal so polynomial regression has a pattern to capture.
    features = features * 2.5
    target = target + 7 * features[:, 0] ** 2
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.25, random_state=17
    )

    models = {
        "Linear": LinearRegression(),
        "Polynomial (degree 2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
        ),
        "Ridge (degree 2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Lasso (degree 2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Lasso(alpha=0.1, max_iter=10000),
        ),
    }
    scores: dict[str, tuple[float, float]] = {}

    print("Regression (lower RMSE and higher R² are better)")
    for name, model in models.items():
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        rmse = float(np.sqrt(mean_squared_error(test_y, prediction)))
        r_squared = float(r2_score(test_y, prediction))
        scores[name] = (rmse, r_squared)
        print(f"{name:24} RMSE={rmse:7.2f}  R²={r_squared:6.3f}")

    assert scores["Polynomial (degree 2)"][0] < scores["Linear"][0]
    return scores


def classification_examples() -> dict[str, float]:
    """Compare Logistic Regression and scaled K-nearest neighbors."""
    features, target = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.2,
        random_state=23,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=23,
        stratify=target,
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=23)
        ),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    scores: dict[str, float] = {}

    print("\nClassification (higher accuracy is better)")
    for name, model in models.items():
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        score = float(accuracy_score(test_y, prediction))
        scores[name] = score
        print(f"{name:24} accuracy={score:.3f}")

    assert all(0.0 <= score <= 1.0 for score in scores.values())
    return scores


def main() -> None:
    regression_examples()
    classification_examples()


if __name__ == "__main__":
    main()