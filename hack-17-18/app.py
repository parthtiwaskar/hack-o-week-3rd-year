"""Serve an interactive browser laboratory for Week 17–18: Deep Learning & LLMs.

Run from repository root: python hack-17-18/app.py
Then open http://127.0.0.1:8001 (or custom port) in a browser.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from week17_18_deep_learning import (
    ConvNet,
    EarlyStopping,
    LinearRegressionModel,
    MiniTransformerClassifier,
    PretrainedFeatureExtractor,
    RegularizedMLP,
    SequenceClassifier,
    TransferLearningModel,
    UnifiedLLMClient,
    demonstrate_tensors_and_autograd,
    get_device,
)

APP_DIRECTORY = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 65536


class LabState:
    """Thread-safe state management for interactive deep learning experiments."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.device = get_device()
        self.llm_client = UnifiedLLMClient(provider="mock")

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "torch_version": torch.__version__,
                "device": str(self.device),
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available(),
                "lab": "Week 17-18: Deep Learning & LLM Laboratory",
            }

    def train_pytorch(self, model_type: str, epochs: int, lr: float) -> dict[str, Any]:
        """Run a PyTorch training loop and return loss history, gradients, and predictions."""
        if not 10 <= epochs <= 500:
            raise ValueError("Epochs must be between 10 and 500.")
        if not 0.0001 <= lr <= 1.0:
            raise ValueError("Learning rate must be between 0.0001 and 1.0.")

        torch.manual_seed(42)
        x_data = torch.linspace(-2, 2, 60).unsqueeze(1)
        y_true = 2.5 * x_data + 1.2 + torch.randn_like(x_data) * 0.12

        if model_type == "linear":
            model = LinearRegressionModel(1, 1)
        else:
            model = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        losses = []
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(x_data)
            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            final_preds = model(x_data).squeeze().tolist()

        # Sample down to 60 points max for smooth web charting
        step = max(1, len(losses) // 60)
        sampled_losses = [losses[i] for i in range(0, len(losses), step)]
        if sampled_losses[-1] != losses[-1]:
            sampled_losses.append(losses[-1])

        autograd_demo = demonstrate_tensors_and_autograd()

        return {
            "model_type": model_type,
            "epochs": epochs,
            "learning_rate": lr,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_history": sampled_losses,
            "sample_x": [round(float(v), 2) for v in x_data.squeeze()[:15]],
            "sample_y_true": [round(float(v), 2) for v in y_true.squeeze()[:15]],
            "sample_y_pred": [round(float(v), 2) for v in final_preds[:15]],
            "autograd_gradients": autograd_demo["gradients"],
            "autograd_loss": autograd_demo["computed_loss"],
        }

    def infer_cnn(self, pattern_type: str) -> dict[str, Any]:
        """Run CNN forward pass and extract intermediate feature representations."""
        torch.manual_seed(42)
        model = ConvNet(in_channels=1, num_classes=3)
        model.eval()

        # Construct distinct synthetic 28x28 patterns (Circle, Cross, Diagonal)
        img = torch.zeros(1, 1, 28, 28)
        if pattern_type == "cross":
            img[0, 0, 10:18, :] = 1.0
            img[0, 0, :, 10:18] = 1.0
        elif pattern_type == "diagonal":
            for i in range(28):
                img[0, 0, i, i] = 1.0
                if i + 1 < 28:
                    img[0, 0, i, i + 1] = 0.8
        else:  # circle / square box
            img[0, 0, 6:22, 6:22] = 0.8
            img[0, 0, 9:19, 9:19] = 0.0

        with torch.no_grad():
            # Pass through Block 1
            feat1 = model.features[0:4](img)  # Conv, BN, ReLU, MaxPool
            # Pass through Block 2
            feat2 = model.features[4:8](feat1)
            # Full forward
            logits = model(img)
            probabilities = F.softmax(logits, dim=1).squeeze().tolist()

        predicted_class = int(np.argmax(probabilities))
        classes = ["Pattern A (Box)", "Pattern B (Cross)", "Pattern C (Diagonal)"]

        return {
            "pattern_type": pattern_type,
            "classes": classes,
            "predicted_class": classes[predicted_class],
            "probabilities": [round(p, 4) for p in probabilities],
            "block1_feature_mean": float(feat1.mean().item()),
            "block2_feature_mean": float(feat2.mean().item()),
            "block1_shape": list(feat1.shape),
            "block2_shape": list(feat2.shape),
        }

    def compare_sequences(self, seq_len: int) -> dict[str, Any]:
        """Compare RNN, LSTM, and GRU sequence processing."""
        seq_len = max(4, min(seq_len, 32))
        dummy_in = torch.randn(1, seq_len, 4)
        comparison = {}

        for cell in ("rnn", "lstm", "gru"):
            model = SequenceClassifier(cell_type=cell, input_size=4, hidden_size=16, num_classes=2)
            model.eval()
            with torch.no_grad():
                logits, last_h = model(dummy_in)
            comparison[cell] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "hidden_vector_norm": round(float(torch.norm(last_h).item()), 3),
                "hidden_preview": [round(float(v), 3) for v in last_h[0, :6]],
                "class_prediction": int(torch.argmax(logits, dim=1).item()),
            }
        return {"sequence_length": seq_len, "models": comparison}

    def test_regularization(
        self, use_dropout: bool, use_batchnorm: bool, use_earlystop: bool
    ) -> dict[str, Any]:
        """Train models with/without regularization to demonstrate prevention of overfitting."""
        torch.manual_seed(42)
        x = torch.randn(180, 16)
        y = (x[:, :2].sum(dim=1, keepdim=True) > 0.0).float()
        x_train, y_train = x[:100], y[:100]
        x_val, y_val = x[100:], y[100:]

        layers: list[nn.Module] = [nn.Linear(16, 32)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(p=0.4))
        layers.append(nn.Linear(32, 1))

        model = nn.Sequential(*layers)
        optimizer = optim.Adam(model.parameters(), lr=0.03)
        criterion = nn.BCEWithLogitsLoss()
        early_stopper = EarlyStopping(patience=5) if use_earlystop else None

        train_losses, val_losses = [], []
        stopped_epoch = None

        for epoch in range(1, 35):
            model.train()
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss = float(criterion(model(x_val), y_val).item())

            train_losses.append(round(float(loss.item()), 4))
            val_losses.append(round(v_loss, 4))

            if early_stopper and early_stopper(v_loss, model):
                stopped_epoch = epoch
                break

        return {
            "use_dropout": use_dropout,
            "use_batchnorm": use_batchnorm,
            "use_earlystop": use_earlystop,
            "epochs_completed": stopped_epoch or 35,
            "early_stopped": stopped_epoch is not None,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
        }

    def inspect_transfer_learning(self, num_classes: int) -> dict[str, Any]:
        """Inspect parameter breakdown for transfer learning model."""
        num_classes = max(2, min(num_classes, 20))
        pretrained = PretrainedFeatureExtractor()
        transfer_model = TransferLearningModel(pretrained, num_custom_classes=num_classes)

        total_params = sum(p.numel() for p in transfer_model.parameters())
        trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "num_classes": num_classes,
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "frozen_percentage": round((frozen_params / total_params) * 100, 1),
        }

    def compute_transformer_attention(self, text: str) -> dict[str, Any]:
        """Tokenize text and compute self-attention matrix."""
        words = [w for w in text.strip().split() if w][:8]
        if not words:
            words = ["attention", "is", "all", "you", "need"]

        # Simple hash token mapping for demo
        token_ids = torch.tensor([[abs(hash(w)) % 40 + 1 for w in words]], dtype=torch.long)
        model = MiniTransformerClassifier(vocab_size=50, embed_dim=8, num_classes=2)
        model.eval()
        with torch.no_grad():
            logits, weights = model(token_ids)
            probs = F.softmax(logits, dim=-1).squeeze().tolist()

        matrix = weights[0].numpy()
        matrix_list = [[round(float(val), 3) for val in row] for row in matrix]

        return {
            "tokens": words,
            "attention_matrix": matrix_list,
            "sentiment_probabilities": [round(p, 4) for p in probs],
            "prediction": "Positive" if probs[1] > probs[0] else "Neutral/Negative",
        }

    def run_llm_query(
        self, provider: str, task: str, prompt: str, api_key: str | None = None
    ) -> dict[str, Any]:
        """Execute text generation, summarization, or Q&A through UnifiedLLMClient."""
        client = UnifiedLLMClient(provider=provider, api_key=api_key)
        if task == "summarize":
            resp = client.summarize(prompt)
        elif task == "qa":
            resp = client.answer_question(context=prompt, question="What are the key technical points?")
        else:
            resp = client.generate(prompt)

        return {
            "provider": resp.provider,
            "model": resp.model,
            "content": resp.content,
            "tokens_used": resp.tokens_used,
            "status": resp.raw_status,
        }


def create_handler(state: LabState) -> type[BaseHTTPRequestHandler]:
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
                    {"error": "Web asset could not be loaded."},
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
                raise ValueError("JSON request body required.")
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
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/api/train/pytorch":
                    res = state.train_pytorch(
                        model_type=body.get("model_type", "linear"),
                        epochs=int(body.get("epochs", 80)),
                        lr=float(body.get("lr", 0.05)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/cnn/infer":
                    res = state.infer_cnn(pattern_type=body.get("pattern_type", "box"))
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/sequence/compare":
                    res = state.compare_sequences(seq_len=int(body.get("seq_len", 8)))
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/regularization/test":
                    res = state.test_regularization(
                        use_dropout=bool(body.get("use_dropout", True)),
                        use_batchnorm=bool(body.get("use_batchnorm", True)),
                        use_earlystop=bool(body.get("use_earlystop", True)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/transfer/setup":
                    res = state.inspect_transfer_learning(
                        num_classes=int(body.get("num_classes", 5))
                    )
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/transformer/attention":
                    res = state.compute_transformer_attention(
                        text=str(body.get("text", "deep learning transformers attention"))
                    )
                    self._send_json(HTTPStatus.OK, res)
                elif self.path == "/api/llm/query":
                    res = state.run_llm_query(
                        provider=body.get("provider", "mock"),
                        task=body.get("task", "generate"),
                        prompt=body.get("prompt", "Explain transfer learning in PyTorch."),
                        api_key=body.get("api_key"),
                    )
                    self._send_json(HTTPStatus.OK, res)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            except ValueError as err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
            except Exception as err:
                logging.exception("Error processing %s", self.path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Internal server error: {err}"},
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format_string % args)

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(LabState()))
    print(f"Deep Learning & LLM Lab running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Deep Learning Lab server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
