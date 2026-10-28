"""Week 5–6: linear algebra and calculus for machine-learning intuition.

Run with: python week5_6_math_foundations.py
Requires numpy.
"""

import numpy as np


def linear_algebra_examples() -> None:
    """Use small vectors and matrices to build geometric intuition."""
    vector = np.array([2.0, -1.0, 3.0])
    direction = np.array([4.0, 0.0, -2.0])
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    point = np.array([1.0, 0.0])

    print("Linear algebra")
    print("Vector:", vector)
    print("Dot product:", np.dot(vector, direction))
    print("Matrix transforms", point, "to", matrix @ point)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors (columns):")
    print(eigenvectors)
    for index, eigenvalue in enumerate(eigenvalues):
        eigenvector = eigenvectors[:, index]
        assert np.allclose(matrix @ eigenvector, eigenvalue * eigenvector)
    print("Each eigenvector keeps its direction under the matrix; its eigenvalue is the scale.")


def sigmoid(value: float) -> float:
    """Squash a real-valued input into the range (0, 1)."""
    return float(1 / (1 + np.exp(-value)))


def loss_for_parameters(
    inputs: np.ndarray, weights: np.ndarray, bias: float, target: float
) -> float:
    """Return half the squared error for one sigmoid neuron."""
    prediction = sigmoid(float(inputs @ weights + bias))
    return 0.5 * (prediction - target) ** 2


def loss_and_gradients(
    inputs: np.ndarray, weights: np.ndarray, bias: float, target: float
) -> tuple[float, float, np.ndarray, float]:
    """Compute a one-example loss and its chain-rule gradients."""
    logit = float(inputs @ weights + bias)
    prediction = sigmoid(logit)
    loss = 0.5 * (prediction - target) ** 2

    d_loss_d_prediction = prediction - target
    d_prediction_d_logit = prediction * (1 - prediction)
    d_loss_d_logit = d_loss_d_prediction * d_prediction_d_logit
    gradient_weights = d_loss_d_logit * inputs
    gradient_bias = d_loss_d_logit
    return loss, prediction, gradient_weights, gradient_bias


def calculus_and_backprop_example() -> None:
    """Connect derivatives, gradients, and the chain rule in one neuron."""
    inputs = np.array([1.5, -2.0])
    weights = np.array([0.2, -0.1])
    bias = 0.05
    target = 1.0

    loss, prediction, gradient_weights, gradient_bias = loss_and_gradients(
        inputs, weights, bias, target
    )
    print("\nCalculus and backprop intuition")
    print(f"Prediction: {prediction:.4f}; target: {target:.1f}; loss: {loss:.4f}")
    print("Chain rule: dLoss/dWeight = dLoss/dPrediction × dPrediction/dLogit × dLogit/dWeight")
    print("Weight gradients:", gradient_weights)
    print(f"Bias gradient: {gradient_bias:.4f}")

    epsilon = 1e-5
    numeric_gradient = np.empty_like(weights)
    for index in range(len(weights)):
        offset = np.zeros_like(weights)
        offset[index] = epsilon
        numeric_gradient[index] = (
            loss_for_parameters(inputs, weights + offset, bias, target)
            - loss_for_parameters(inputs, weights - offset, bias, target)
        ) / (2 * epsilon)
    assert np.allclose(gradient_weights, numeric_gradient, atol=1e-7)

    learning_rate = 0.5
    updated_weights = weights - learning_rate * gradient_weights
    updated_bias = bias - learning_rate * gradient_bias
    updated_loss = loss_for_parameters(inputs, updated_weights, updated_bias, target)
    assert updated_loss < loss
    print(f"Numerical derivative check passed; one gradient step lowers loss to {updated_loss:.4f}.")


def main() -> None:
    linear_algebra_examples()
    calculus_and_backprop_example()


if __name__ == "__main__":
    main()