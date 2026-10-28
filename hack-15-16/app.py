"""Serve an interactive browser demo for the Week 15–16 neural-network lesson.

Run from the repository root with: python hack-15-16/app.py
Then open http://127.0.0.1:8000 in a browser.
Requires the packages in hack-15-16/requirements.txt.
"""

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from week15_16_neural_networks import NeuralNetwork


APP_DIRECTORY = Path(__file__).resolve().parent
FEATURES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
TARGETS = np.array([0, 1, 1, 0])
MAX_REQUEST_BYTES = 4096
MAX_EPOCHS = 10_000


class DemoState:
    """Keep the latest trained model safe across concurrent HTTP requests."""

    def __init__(self) -> None:
        self.model: NeuralNetwork | None = None
        self.optimizer: str | None = None
        self.epochs: int | None = None
        self.lock = Lock()

    def train(self, optimizer: str, epochs: int) -> dict[str, Any]:
        """Train XOR and return a compact learning history and predictions."""
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("Choose either SGD or Adam.")
        if not 100 <= epochs <= MAX_EPOCHS:
            raise ValueError(f"Epochs must be between 100 and {MAX_EPOCHS}.")

        learning_rate = 0.1 if optimizer == "sgd" else 0.03
        model = NeuralNetwork(
            input_size=2,
            hidden_size=8,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=17,
        )
        history = model.fit(FEATURES, TARGETS, epochs=epochs, batch_size=2)
        with self.lock:
            self.model = model
            self.optimizer = optimizer
            self.epochs = epochs

        sample_indexes = np.linspace(
            0, len(history) - 1, num=min(100, len(history)), dtype=int
        )
        probabilities = model.predict_proba(FEATURES).ravel()
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "optimizer": optimizer,
            "epochs": epochs,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "loss_history": [history[index] for index in sample_indexes],
            "accuracy": float(np.mean(predictions == TARGETS)),
            "predictions": [
                {
                    "inputs": inputs.astype(int).tolist(),
                    "target": int(target),
                    "probability": float(probability),
                    "prediction": int(prediction),
                }
                for inputs, target, probability, prediction in zip(
                    FEATURES, TARGETS, probabilities, predictions
                )
            ],
        }

    def predict(self, first_input: Any, second_input: Any) -> dict[str, Any]:
        """Return the active model's probability for an input pair."""
        try:
            inputs = [float(first_input), float(second_input)]
        except (TypeError, ValueError) as error:
            raise ValueError("Both inputs must be either 0 or 1.") from error
        if any(value not in {0.0, 1.0} for value in inputs):
            raise ValueError("Both inputs must be either 0 or 1.")

        with self.lock:
            if self.model is None:
                raise ValueError("Train a model before asking it for a prediction.")
            probability = float(
                self.model.predict_proba(np.array([inputs], dtype=float))[0, 0]
            )
            optimizer = self.optimizer
        return {
            "inputs": [int(value) for value in inputs],
            "probability": probability,
            "prediction": int(probability >= 0.5),
            "optimizer": optimizer,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "trained": self.model is not None,
                "optimizer": self.optimizer,
                "epochs": self.epochs,
            }


def create_handler(state: DemoState) -> type[BaseHTTPRequestHandler]:
    """Create a request handler connected to this server's model state."""

    class RequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("A JSON request body is required.")
            try:
                body_length = int(content_length)
            except ValueError as error:
                raise ValueError("Content-Length must be an integer.") from error
            if body_length < 1 or body_length > MAX_REQUEST_BYTES:
                raise ValueError("The JSON request body has an invalid size.")
            try:
                data = json.loads(self.rfile.read(body_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Request body must contain valid JSON.") from error
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")
            return data

        def _send_static_file(
            self, file_path: Path, content_type: str
        ) -> None:
            try:
                content = file_path.read_bytes()
            except OSError:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "The web app asset could not be read."},
                )
                logging.exception("Could not read web app asset %s", file_path)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "index.html",
                    "text/html; charset=utf-8",
                )
            elif self.path == "/styles.css":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "styles.css",
                    "text/css; charset=utf-8",
                )
            elif self.path == "/app.js":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "app.js",
                    "text/javascript; charset=utf-8",
                )
            elif self.path == "/api/status":
                self._send_json(HTTPStatus.OK, state.status())
            elif self.path in {"/favicon.ico", "/favicon.png"}:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            elif self.path.startswith("/.well-known/"):
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

        def do_POST(self) -> None:
            try:
                data = self._read_json()
                if self.path == "/api/train":
                    optimizer = data.get("optimizer", "adam")
                    epochs = data.get("epochs", 3000)
                    if not isinstance(optimizer, str):
                        raise ValueError("Optimizer must be either SGD or Adam.")
                    if isinstance(epochs, bool) or not isinstance(epochs, int):
                        raise ValueError("Epochs must be a whole number.")
                    self._send_json(HTTPStatus.OK, state.train(optimizer, epochs))
                elif self.path == "/api/predict":
                    self._send_json(
                        HTTPStatus.OK,
                        state.predict(data.get("first"), data.get("second")),
                    )
                else:
                    self._send_json(
                        HTTPStatus.NOT_FOUND, {"error": "Route not found."}
                    )
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception:
                logging.exception("Request failed for %s", self.path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "An unexpected server error occurred."},
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format_string % args)

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (defaults to localhost)",
    )
    parser.add_argument("--port", type=int, default=8000, help="port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(DemoState()))
    print(f"Neural network lab running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping neural network lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
