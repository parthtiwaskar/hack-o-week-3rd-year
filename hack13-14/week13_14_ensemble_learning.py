"""Week 13–14: Ensemble Methods (Bagging, Boosting, XGBoost, LightGBM),
Bias-Variance Trade-off, Overfitting/Underfitting, and L1/L2 Regularization.

Run with: python hack13-14/week13_14_ensemble_learning.py
Requires numpy, scikit-learn, xgboost, and lightgbm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import lightgbm as lgb
import xgboost as xgb


# ============================================================================
# 1. Ensemble Methods: Bagging vs. Boosting (XGBoost & LightGBM)
# ============================================================================

def compare_ensemble_methods(
    n_samples: int = 600, n_features: int = 10, noise_ratio: float = 0.15
) -> dict[str, dict[str, Any]]:
    """Compare single Decision Tree, Bagging (Random Forest), and Boosting (GradientBoosting, XGBoost, LightGBM)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        flip_y=noise_ratio,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Single Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Bagging (Random Forest)": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=60,
            learning_rate=0.1,
            random_state=42,
        ),
        "Gradient Boosting (sklearn)": GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            verbose=-1,
            random_state=42,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))

        results[name] = {
            "test_accuracy": acc,
            "test_auc": auc,
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        }

    # Verify that ensemble methods outperform a single unpruned decision tree
    rf_acc = results["Bagging (Random Forest)"]["test_accuracy"]
    xgb_acc = results["XGBoost"]["test_accuracy"]
    dt_acc = results["Single Decision Tree"]["test_accuracy"]
    assert max(rf_acc, xgb_acc) >= dt_acc, "Ensemble should outperform single tree"

    return results


# ============================================================================
# 2. Bias-Variance Decomposition
# ============================================================================

def compute_bias_variance_decomposition(
    n_trials: int = 50, n_test: int = 80
) -> dict[str, Any]:
    """Decompose Mean Squared Error into (Bias² + Variance + Noise) across tree depths.
    
    True DGP: y = sin(pi * x) + noise(sigma=0.3)
    """
    rng = np.random.default_rng(42)
    x_test = np.linspace(-1.0, 1.0, n_test).reshape(-1, 1)
    true_y_test = np.sin(np.pi * x_test).ravel()
    noise_variance = 0.3**2

    depths = [1, 2, 4, 8, 14]
    decomposition: dict[int, dict[str, float]] = {}

    for depth in depths:
        predictions = np.zeros((n_trials, n_test))

        for trial in range(n_trials):
            # Generate new training draw
            x_train = rng.uniform(-1.0, 1.0, size=(50, 1))
            y_train = np.sin(np.pi * x_train).ravel() + rng.normal(0, 0.3, size=50)

            model = DecisionTreeRegressor(max_depth=depth, random_state=trial)
            model.fit(x_train, y_train)
            predictions[trial, :] = model.predict(x_test)

        # Expected prediction across all trained models
        mean_prediction = np.mean(predictions, axis=0)

        # Bias²: E[(f_hat(x) - f(x))²]
        bias_squared = float(np.mean((mean_prediction - true_y_test) ** 2))

        # Variance: E[(f_hat(x) - E[f_hat(x)])²]
        variance = float(np.mean(np.var(predictions, axis=0)))

        # Total Expected Error = Bias² + Variance + Noise
        total_mse = bias_squared + variance + noise_variance

        decomposition[depth] = {
            "bias_squared": round(bias_squared, 4),
            "variance": round(variance, 4),
            "noise": round(noise_variance, 4),
            "total_expected_mse": round(total_mse, 4),
        }

    # Low depth (1): High bias, low variance (underfitting)
    # High depth (14): Low bias, high variance (overfitting)
    assert decomposition[1]["bias_squared"] > decomposition[14]["bias_squared"]
    assert decomposition[1]["variance"] < decomposition[14]["variance"]

    return {
        "depths": depths,
        "decomposition": decomposition,
        "sweet_spot_depth": min(
            decomposition.keys(), key=lambda d: decomposition[d]["total_expected_mse"]
        ),
    }


# ============================================================================
# 3. Overfitting vs. Underfitting: Identification & Prevention
# ============================================================================

def analyze_overfitting_underfitting() -> dict[str, Any]:
    """Inspect train vs. validation divergence across model capacity."""
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=5,
        n_redundant=3,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )

    depth_range = list(range(1, 16))
    train_scores: list[float] = []
    val_scores: list[float] = []

    for depth in depth_range:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(round(float(accuracy_score(y_train, clf.predict(X_train))), 4))
        val_scores.append(round(float(accuracy_score(y_val, clf.predict(X_val))), 4))

    optimal_depth = depth_range[int(np.argmax(val_scores))]

    # Underfitting check: Depth 1 has low train & val accuracy
    # Overfitting check: Depth 15 has 100% train accuracy, but dropped val accuracy
    generalization_gap_deep = train_scores[-1] - val_scores[-1]
    generalization_gap_opt = train_scores[optimal_depth - 1] - val_scores[optimal_depth - 1]
    assert generalization_gap_deep > generalization_gap_opt

    return {
        "depth_range": depth_range,
        "train_accuracies": train_scores,
        "validation_accuracies": val_scores,
        "optimal_depth": optimal_depth,
        "best_val_accuracy": max(val_scores),
        "deep_overfit_gap": round(generalization_gap_deep, 4),
    }


# ============================================================================
# 4. Regularization: L1 (Lasso) vs. L2 (Ridge) in Ensembles & Linear Models
# ============================================================================

def demonstrate_l1_l2_regularization() -> dict[str, Any]:
    """Compare L1 (feature sparsity/selection) and L2 (weight shrinkage).
    
    Demonstrates:
    - Linear: Lasso (L1) vs Ridge (L2)
    - Boosting: XGBoost with L1 (reg_alpha) and L2 (reg_lambda)
    """
    # 5 informative features + 15 pure noise features
    rng = np.random.default_rng(42)
    n_samples = 250
    X_informative = rng.normal(size=(n_samples, 5))
    X_noise = rng.normal(size=(n_samples, 15))
    X = np.hstack([X_informative, X_noise])
    # True target depends ONLY on the first 5 features
    true_weights = np.array([3.5, -2.0, 1.8, -1.2, 0.9])
    y = X_informative @ true_weights + rng.normal(0, 0.5, size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. Linear L1 vs L2
    unreg = LinearRegression().fit(X_train, y_train)
    lasso_l1 = Lasso(alpha=0.25).fit(X_train, y_train)
    ridge_l2 = Ridge(alpha=10.0).fit(X_train, y_train)

    # L1 should zero out most noise features
    lasso_zeroed_features = int(np.sum(np.isclose(lasso_l1.coef_[5:], 0.0, atol=1e-3)))
    ridge_non_zero_noise = int(np.sum(~np.isclose(ridge_l2.coef_[5:], 0.0, atol=1e-3)))

    # 2. XGBoost L1/L2 Regularization
    xgb_unreg = xgb.XGBRegressor(
        n_estimators=60, max_depth=4, reg_alpha=0.0, reg_lambda=0.0, random_state=42
    ).fit(X_train, y_train)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=60, max_depth=4, reg_alpha=1.5, reg_lambda=2.0, random_state=42
    ).fit(X_train, y_train)

    mse_unreg = float(mean_squared_error(y_test, xgb_unreg.predict(X_test)))
    mse_reg = float(mean_squared_error(y_test, xgb_reg.predict(X_test)))

    return {
        "l1_sparsity": {
            "noise_features_count": 15,
            "lasso_zeroed_features": lasso_zeroed_features,
            "ridge_zeroed_features": 15 - ridge_non_zero_noise,
            "lasso_test_mse": round(float(mean_squared_error(y_test, lasso_l1.predict(X_test))), 4),
            "ridge_test_mse": round(float(mean_squared_error(y_test, ridge_l2.predict(X_test))), 4),
        },
        "xgboost_regularization": {
            "unregularized_test_mse": round(mse_unreg, 4),
            "regularized_test_mse": round(mse_reg, 4),
            "improvement": round(((mse_unreg - mse_reg) / mse_unreg) * 100, 2),
            "unreg_importances": [round(float(v), 3) for v in xgb_unreg.feature_importances_[:8]],
            "reg_importances": [round(float(v), 3) for v in xgb_reg.feature_importances_[:8]],
        },
    }


# ============================================================================
# Main Demonstration Runner
# ============================================================================

def run_all_demonstrations() -> None:
    print("=" * 72)
    print("  WEEK 13–14: ENSEMBLE METHODS, BIAS-VARIANCE TRADEOFF & REGULARIZATION")
    print("=" * 72)

    # 1. Ensemble Methods
    print("\n[1] Ensemble Methods: Bagging vs. Boosting (XGBoost & LightGBM)")
    res_ensembles = compare_ensemble_methods()
    for name, metrics in res_ensembles.items():
        print(
            f"  • {name:28} | Test Acc: {metrics['test_accuracy']:.2%} | Test AUC: {metrics['test_auc']:.3f} | Train Acc: {metrics['train_accuracy']:.2%}"
        )

    # 2. Bias-Variance Decomposition
    print("\n[2] Bias-Variance Tradeoff Analysis")
    bv_res = compute_bias_variance_decomposition()
    print("  Depth | Bias²     | Variance  | Noise    | Total Expected MSE")
    print("  -------------------------------------------------------------")
    for depth in bv_res["depths"]:
        d = bv_res["decomposition"][depth]
        print(f"  {depth:5} | {d['bias_squared']:8.4f} | {d['variance']:8.4f} | {d['noise']:8.4f} | {d['total_expected_mse']:8.4f}")
    print(f"  Optimal Sweet Spot Depth: {bv_res['sweet_spot_depth']} (minimizes Bias² + Variance)")

    # 3. Overfitting / Underfitting
    print("\n[3] Overfitting & Underfitting Identification")
    fit_res = analyze_overfitting_underfitting()
    print(f"  • Underfitting (Depth=1) : Train Acc = {fit_res['train_accuracies'][0]:.2%}, Val Acc = {fit_res['validation_accuracies'][0]:.2%}")
    print(f"  • Optimal Sweet Spot ({fit_res['optimal_depth']}) : Train Acc = {fit_res['train_accuracies'][fit_res['optimal_depth']-1]:.2%}, Val Acc = {fit_res['best_val_accuracy']:.2%}")
    print(f"  • Overfitting (Depth=15) : Train Acc = {fit_res['train_accuracies'][-1]:.2%}, Val Acc = {fit_res['validation_accuracies'][-1]:.2%} (Gap = {fit_res['deep_overfit_gap']:.2%})")

    # 4. Regularization L1/L2
    print("\n[4] Regularization: L1 (Sparsity) vs. L2 (Weight Shrinkage)")
    reg_res = demonstrate_l1_l2_regularization()
    l1_data = reg_res["l1_sparsity"]
    print(f"  • Lasso (L1): Successfully zeroed {l1_data['lasso_zeroed_features']}/15 noise features.")
    xgb_reg_data = reg_res["xgboost_regularization"]
    print(f"  • XGBoost without regularization MSE : {xgb_reg_data['unregularized_test_mse']}")
    print(f"  • XGBoost with L1/L2 regularized MSE : {xgb_reg_data['regularized_test_mse']} ({xgb_reg_data['improvement']}% improvement on noisy test data)")

    print("\n" + "=" * 72)
    print("  ALL DEMONSTRATIONS AND TESTS PASSED SUCCESSFULLY!")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true", help="run tests and exit")
    args = parser.parse_args()
    run_all_demonstrations()


if __name__ == "__main__":
    main()
