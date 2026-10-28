"""Serve an interactive browser laboratory for Week 13–14:
Ensemble Methods, Bias-Variance Tradeoff, and Regularization.

Run from repository root: python hack13-14/app.py --port 8002
Then open http://127.0.0.1:8002 in a browser.
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from week13_14_ensemble_learning import (
    analyze_overfitting_underfitting,
    compare_ensemble_methods,
    compute_bias_variance_decomposition,
    demonstrate_l1_l2_regularization,
)

APP_DIRECTORY = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 65536


class EnsembleLabState:
    """Thread-safe lab state for Week 13-14 ensemble experiments."""

    def __init__(self) -> None:
        self.lock = Lock()

    def get_status(self) -> dict[str, Any]:
        return {
            "xgboost_version": xgb.__version__,
            "lightgbm_version": lgb.__version__,
            "lab": "Week 13-14: Ensemble Methods & Regularization Lab",
        }

    def benchmark_ensembles(self, n_samples: int, noise_ratio: float) -> dict[str, Any]:
        with self.lock:
            n_samples = max(200, min(n_samples, 2000))
            noise_ratio = max(0.0, min(noise_ratio, 0.4))
            res = compare_ensemble_methods(n_samples=n_samples, noise_ratio=noise_ratio)
            return {"samples": n_samples, "noise_ratio": noise_ratio, "models": res}

    def get_bias_variance(self) -> dict[str, Any]:
        with self.lock:
            return compute_bias_variance_decomposition(n_trials=40, n_test=60)

    def get_overfitting_curve(self) -> dict[str, Any]:
        with self.lock:
            return analyze_overfitting_underfitting()

    def get_regularization_data(self) -> dict[str, Any]:
        with self.lock:
            return demonstrate_l1_l2_regularization()


def create_handler(state: EnsembleLabState) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_static_file(self, file_path: Path, content_type: str) -> None:
            try:
                content = file_path.read_bytes()
            except OSError:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Asset could not be read."},
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def _read_json(self) -> dict[str, Any]:
            content_length = self.headers.get("Content-Length")
            if not content_length:
                raise ValueError("JSON body required.")
            length = int(content_length)
            if length > MAX_REQUEST_BYTES:
                raise ValueError("Request body too large.")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "index.html", "text/html; charset=utf-8"
                )
            elif self.path == "/styles.css":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "styles.css", "text/css; charset=utf-8"
                )
            elif self.path == "/app.js":
                self._send_static_file(
                    APP_DIRECTORY / "static" / "app.js", "text/javascript; charset=utf-8"
                )
            elif self.path in {"/favicon.ico", "/favicon.png"}:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            elif self.path == "/api/status":
                self._send_json(HTTPStatus.OK, state.get_status())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/api/ensemble/benchmark":
                    res = state.benchmark_ensembles(
                        n_samples=int(body.get("n_samples", 600)),
                        noise_ratio=float(body.get("noise_ratio", 0.15)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/bias-variance/decompose":
                    res = state.get_bias_variance()
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/overfitting/curve":
                    res = state.get_overfitting_curve()
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/regularization/compare":
                    res = state.get_regularization_data()
                    self._send_json(HTTPStatus.OK, res)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            except ValueError as err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
            except Exception as err:
                logging.exception("Error during request to %s", self.path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Server error: {err}"},
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format_string % args)

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(EnsembleLabState()))
    print(f"Ensemble & Regularization Lab running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Ensemble Lab server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
