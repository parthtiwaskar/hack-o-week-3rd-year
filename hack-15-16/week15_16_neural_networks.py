"""Week 15–16: neural-network basics and model deployment with NumPy.

Run with: python week15_16_neural_networks.py
Models are written to ./week15_16_models by default. Use --output-dir to
choose another directory.
Requires numpy and joblib.
"""

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Map logits to probabilities while avoiding exponential overflow."""
    clipped = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-clipped))


def relu(values: np.ndarray) -> np.ndarray:
    """Keep positive values and set negative values to zero."""
    return np.maximum(0, values)


def tanh(values: np.ndarray) -> np.ndarray:
    """Squash values into the range (-1, 1)."""
    return np.tanh(values)


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert a row of logits into class probabilities."""
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def binary_cross_entropy(
    targets: np.ndarray, probabilities: np.ndarray
) -> float:
    """Calculate binary cross-entropy, clipping logs away from zero."""
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return float(
        -np.mean(
            targets * np.log(probabilities)
            + (1 - targets) * np.log(1 - probabilities)
        )
    )


def mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate the average squared prediction error."""
    return float(np.mean((targets - predictions) ** 2))


def perceptron(inputs: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return binary predictions from a thresholded linear unit."""
    return (inputs @ weights + bias >= 0).astype(int)


class NeuralNetwork:
    """A one-hidden-layer binary classifier trained with backpropagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 8,
        learning_rate: float = 0.03,
        optimizer: str = "adam",
        seed: int = 17,
    ) -> None:
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        rng = np.random.default_rng(seed)
        self.parameters = {
            "W1": rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size)),
            "b1": np.zeros((1, hidden_size)),
            "W2": rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, 1)),
            "b2": np.zeros((1, 1)),
        }
        self._adam_first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self._adam_second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self._optimizer_step = 0

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = relu(features @ self.parameters["W1"] + self.parameters["b1"])
        probabilities = sigmoid(hidden @ self.parameters["W2"] + self.parameters["b2"])
        return hidden, probabilities

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a probability for the positive class for each input row."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(f"features must have shape (n, {self.input_size})")
        if len(features) == 0:
            raise ValueError("features must contain at least one row")
        return self._forward(features)[1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return binary class predictions using a 0.5 threshold."""
        return (self.predict_proba(features) >= 0.5).astype(int)

    def _update_parameters(self, gradients: dict[str, np.ndarray]) -> None:
        if self.optimizer == "sgd":
            for name, gradient in gradients.items():
                self.parameters[name] -= self.learning_rate * gradient
            return

        self._optimizer_step += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        for name, gradient in gradients.items():
            first_moment = self._adam_first_moment[name]
            second_moment = self._adam_second_moment[name]
            first_moment *= beta1
            first_moment += (1 - beta1) * gradient
            second_moment *= beta2
            second_moment += (1 - beta2) * gradient**2
            corrected_first = first_moment / (1 - beta1**self._optimizer_step)
            corrected_second = second_moment / (1 - beta2**self._optimizer_step)
            self.parameters[name] -= (
                self.learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + epsilon)
            )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 3000,
        batch_size: int = 4,
        seed: int = 29,
    ) -> list[float]:
        """Train using mini-batch gradient descent and return epoch losses."""
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(-1, 1)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(f"features must have shape (n, {self.input_size})")
        if len(features) == 0 or len(features) != len(targets):
            raise ValueError("features and targets must have the same nonzero row count")
        if not np.all(np.isin(targets, [0, 1])):
            raise ValueError("targets must contain only binary labels 0 and 1")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(features))
            for start in range(0, len(features), batch_size):
                indices = order[start : start + batch_size]
                batch_features = features[indices]
                batch_targets = targets[indices]
                hidden, probabilities = self._forward(batch_features)

                output_gradient = (probabilities - batch_targets) / len(indices)
                gradients = {
                    "W2": hidden.T @ output_gradient,
                    "b2": np.sum(output_gradient, axis=0, keepdims=True),
                }
                hidden_gradient = output_gradient @ self.parameters["W2"].T
                hidden_gradient *= hidden > 0
                gradients["W1"] = batch_features.T @ hidden_gradient
                gradients["b1"] = np.sum(hidden_gradient, axis=0, keepdims=True)
                self._update_parameters(gradients)

            epoch_probabilities = self.predict_proba(features)
            history.append(binary_cross_entropy(targets, epoch_probabilities))
        return history


def save_with_pickle(model: NeuralNetwork, path: Path) -> None:
    """Serialize a model with Python's standard-library pickle format."""
    with path.open("wb") as model_file:
        pickle.dump(model, model_file)


def load_with_pickle(path: Path) -> NeuralNetwork:
    """Load a pickle model. Only load files from trusted sources."""
    with path.open("rb") as model_file:
        return pickle.load(model_file)


def save_with_joblib(model: NeuralNetwork, path: Path) -> None:
    """Serialize a model with joblib, which is useful for NumPy-heavy objects."""
    joblib.dump(model, path)


def load_with_joblib(path: Path) -> NeuralNetwork:
    """Load a joblib model. Only load files from trusted sources."""
    return joblib.load(path)


def demonstrate_basics() -> None:
    """Show a perceptron, common activation functions, and loss functions."""
    truth_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_predictions = perceptron(
        truth_table, np.array([1.0, 1.0]), bias=-1.5
    )
    logits = np.array([[-1.0, 0.5, 2.0]])

    print("Perceptron: a thresholded weighted sum (AND truth table)")
    print(np.column_stack((truth_table, and_predictions)))
    print("\nActivation functions for logits [-1.0, 0.5, 2.0]")
    print("Sigmoid:", sigmoid(logits))
    print("ReLU:", relu(logits))
    print("Tanh:", tanh(logits))
    print("Softmax:", softmax(logits))
    print("\nLoss functions")
    print(
        "Binary cross-entropy:",
        binary_cross_entropy(np.array([[0], [1]]), np.array([[0.2], [0.8]])),
    )
    print(
        "Mean squared error:",
        mean_squared_error(np.array([0, 1]), np.array([0.2, 0.8])),
    )
    print("\nNeural-network training")
    print("Forward: inputs -> ReLU(X @ W1 + b1) -> sigmoid(H @ W2 + b2)")
    print("Backward: binary-cross-entropy gradients flow through the layers by the chain rule.")
    print("SGD applies each mini-batch gradient; Adam adapts updates using gradient moments.")


def train_xor_model(optimizer: str) -> tuple[NeuralNetwork, list[float]]:
    """Train a small neural network to learn the non-linear XOR pattern."""
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([0, 1, 1, 0])
    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        learning_rate=0.03 if optimizer == "adam" else 0.1,
        optimizer=optimizer,
        seed=17,
    )
    history = model.fit(features, targets, epochs=3000, batch_size=2)
    return model, history


def demonstrate_training_and_deployment(output_dir: Path) -> None:
    """Compare optimizers, then save and reload a trained model both ways."""
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([0, 1, 1, 0])

    print("\nTraining a neural network from scratch on XOR")
    trained_models = {}
    for optimizer in ("sgd", "adam"):
        model, history = train_xor_model(optimizer)
        predictions = model.predict(features).ravel()
        accuracy = float(np.mean(predictions == targets))
        print(
            f"{optimizer.upper()}: loss {history[0]:.4f} -> {history[-1]:.4f}; "
            f"accuracy {accuracy:.0%}; predictions {predictions.tolist()}"
        )
        assert history[-1] < history[0]
        assert accuracy == 1.0
        trained_models[optimizer] = model

    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / "xor_model.pkl"
    joblib_path = output_dir / "xor_model.joblib"
    model = trained_models["adam"]
    save_with_pickle(model, pickle_path)
    save_with_joblib(model, joblib_path)

    original_predictions = model.predict_proba(features)
    pickle_predictions = load_with_pickle(pickle_path).predict_proba(features)
    joblib_predictions = load_with_joblib(joblib_path).predict_proba(features)
    assert np.allclose(pickle_predictions, original_predictions)
    assert np.allclose(joblib_predictions, original_predictions)
    print(f"\nSaved and reloaded model with pickle: {pickle_path}")
    print(f"Saved and reloaded model with joblib: {joblib_path}")
    print("Reloaded predictions match the original model.")
    print("Security note: never load pickle/joblib files from untrusted sources.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("week15_16_models"),
        help="directory for the saved pickle and joblib models",
    )
    args = parser.parse_args()

    demonstrate_basics()
    demonstrate_training_and_deployment(args.output_dir)


if __name__ == "__main__":
    main()
