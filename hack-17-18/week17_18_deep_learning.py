"""Week 17–18: Deep Learning with PyTorch, CNNs, RNNs/LSTMs, Regularization,
Transfer Learning, Transformers, and LLM APIs.

Run with: python hack-17-18/week17_18_deep_learning.py
Requires torch, torchvision, numpy, and requests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# Device Setup
# ============================================================================

def get_device() -> torch.device:
    """Select the best available compute device (MPS on Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# 1. PyTorch Basics: Tensors, Autograd, and Training Loop
# ============================================================================

def demonstrate_tensors_and_autograd() -> dict[str, Any]:
    """Demonstrate tensor creation, reshaping, operations, and autograd gradients."""
    # Tensor creation
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.randn(2, 3, generator=torch.manual_seed(42))
    from_np = torch.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Autograd: track computation graph
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    # y = 3 * x^2 + 2 * x + 1
    y = 3 * (x**2) + 2 * x + 1
    loss = y.sum()
    loss.backward()
    # dy/dx = 6 * x + 2 => for [2, 3] -> [14, 20]
    expected_grad = torch.tensor([14.0, 20.0])
    assert torch.allclose(x.grad, expected_grad), f"Grad mismatch: {x.grad} vs {expected_grad}"

    return {
        "zeros_shape": list(zeros.shape),
        "ones_shape": list(ones.shape),
        "rand_sample": rand.tolist(),
        "from_numpy": from_np.tolist(),
        "x": x.detach().tolist(),
        "computed_loss": float(loss.item()),
        "gradients": x.grad.tolist(),
    }


class LinearRegressionModel(nn.Module):
    """Simple linear regression model: y = X @ W + b."""

    def __init__(self, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_model(
    epochs: int = 100, lr: float = 0.05
) -> tuple[LinearRegressionModel, list[float]]:
    """Train a linear regression model using PyTorch's training loop."""
    torch.manual_seed(42)
    # Synthetic target: y = 2.5 * x + 1.2 + noise
    x_train = torch.linspace(-2, 2, steps=60).unsqueeze(1)
    true_w = 2.5
    true_b = 1.2
    y_train = true_w * x_train + true_b + torch.randn_like(x_train) * 0.15

    model = LinearRegressionModel(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    assert history[-1] < history[0], "Loss failed to decrease"
    return model, history


# ============================================================================
# 2. Convolutional Neural Networks (CNNs)
# ============================================================================

class ConvNet(nn.Module):
    """A clean CNN for image classification with convolutions, pooling, and dropout."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 16, H/2, W/2)
            # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, H/4, W/4)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits


def test_cnn_architecture() -> dict[str, Any]:
    """Test CNN forward pass and shape transformations."""
    torch.manual_seed(42)
    model = ConvNet(in_channels=1, num_classes=3)
    model.eval()

    # Synthetic batch of 4 grayscale images (4, 1, 28, 28)
    dummy_images = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        logits = model(dummy_images)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    assert logits.shape == (4, 3)
    return {
        "input_shape": list(dummy_images.shape),
        "output_shape": list(logits.shape),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


# ============================================================================
# 3. Recurrent Neural Networks: RNN, LSTM, and GRU
# ============================================================================

class SequenceClassifier(nn.Module):
    """Sequence classifier supporting 'rnn', 'lstm', or 'gru' backbones."""

    def __init__(
        self,
        cell_type: str = "lstm",
        input_size: int = 4,
        hidden_size: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type.lower()
        self.hidden_size = hidden_size

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown cell_type '{cell_type}', choose rnn/lstm/gru")

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_length, input_size)
        if self.cell_type == "lstm":
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
        # Use last timestep hidden state for classification
        last_hidden = out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits, last_hidden


def compare_sequence_models() -> dict[str, Any]:
    """Compare parameter counts and output shapes for RNN, LSTM, and GRU."""
    batch_size, seq_len, feat_dim = 2, 8, 4
    dummy_seq = torch.randn(batch_size, seq_len, feat_dim)
    results = {}

    for cell in ("rnn", "lstm", "gru"):
        model = SequenceClassifier(cell_type=cell, input_size=feat_dim, hidden_size=16, num_classes=2)
        model.eval()
        with torch.no_grad():
            logits, last_h = model(dummy_seq)
        param_count = sum(p.numel() for p in model.parameters())
        results[cell] = {
            "parameters": param_count,
            "logits_shape": list(logits.shape),
            "hidden_shape": list(last_h.shape),
        }

    return results


# ============================================================================
# 4. Regularization: Dropout, Batch Normalization, and Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stops training when validation loss stops improving after patience epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0
        self.early_stop = False
        self.best_state: dict[str, torch.Tensor] | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class RegularizedMLP(nn.Module):
    """Multilayer perceptron with Dropout and Batch Normalization."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demonstrate_regularization() -> dict[str, Any]:
    """Train with EarlyStopping to show how overfitting is checked."""
    torch.manual_seed(42)
    # Generate synthetic dataset with noisy features
    x = torch.randn(200, 16)
    y = (x[:, :2].sum(dim=1, keepdim=True) > 0.0).float()

    x_train, y_train = x[:120], y[:120]
    x_val, y_val = x[120:], y[120:]

    model = RegularizedMLP(in_features=16, hidden_dim=32, dropout_p=0.4)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=4)

    stopped_epoch = None
    train_losses, val_losses = [], []
    for epoch in range(1, 40):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(x_val), y_val).item()

        train_losses.append(float(loss.item()))
        val_losses.append(float(v_loss))

        if early_stopper(v_loss, model):
            stopped_epoch = epoch
            early_stopper.restore_best_weights(model)
            break

    return {
        "epochs_run": stopped_epoch or 40,
        "best_val_loss": early_stopper.best_loss,
        "early_stopped": early_stopper.early_stop,
        "final_train_loss": train_losses[-1],
    }


# ============================================================================
# 5. Transfer Learning Basics
# ============================================================================

class PretrainedFeatureExtractor(nn.Module):
    """Simulated pretrained backbone (or lightweight torchvision model)."""

    def __init__(self) -> None:
        super().__init__()
        # Pretrained representation backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        # Pretrained classifier head
        self.head = nn.Linear(16 * 2 * 2, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(torch.flatten(feat, 1))


class TransferLearningModel(nn.Module):
    """Model that freezes a pretrained backbone and replaces the classification head."""

    def __init__(self, pretrained_net: nn.Module, num_custom_classes: int = 5) -> None:
        super().__init__()
        # 1. Reuse backbone
        self.backbone = getattr(pretrained_net, "backbone", pretrained_net)
        # 2. Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. Attach a custom head for new target task
        self.custom_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_custom_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.custom_head(features)


def demonstrate_transfer_learning() -> dict[str, Any]:
    """Demonstrate parameter freezing and custom head adaptation."""
    pretrained = PretrainedFeatureExtractor()
    transfer_model = TransferLearningModel(pretrained, num_custom_classes=4)

    total_params = sum(p.numel() for p in transfer_model.parameters())
    trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Verify frozen layers don't compute gradients
    dummy_input = torch.randn(2, 3, 16, 16)
    out = transfer_model(dummy_input)
    loss = out.sum()
    loss.backward()

    # Backbone parameters should have None or zero grad
    for param in transfer_model.backbone.parameters():
        assert param.grad is None, "Frozen backbone parameter should not receive gradients!"

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "output_shape": list(out.shape),
    }


# ============================================================================
# 6. Hugging Face Transformers Basics & Self-Attention
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MiniTransformerClassifier(nn.Module):
    """Lightweight self-contained Transformer encoder for text classification."""

    def __init__(
        self, vocab_size: int = 100, embed_dim: int = 16, num_classes: int = 2
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = ScaledDotProductAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, seq_len, embed_dim)
        embeddings = self.embedding(token_ids)
        attended, weights = self.attention(embeddings, embeddings, embeddings)
        # Average pooling across sequence tokens
        pooled = attended.mean(dim=1)
        logits = self.fc(pooled)
        return logits, weights


def demonstrate_transformer_attention() -> dict[str, Any]:
    """Demonstrate self-attention mechanism and classification on sample token sequences."""
    torch.manual_seed(42)
    model = MiniTransformerClassifier(vocab_size=50, embed_dim=8, num_classes=2)
    model.eval()

    # Batch of 2 sentences, each 5 tokens
    sample_tokens = torch.tensor([[5, 12, 19, 4, 1], [2, 8, 44, 15, 0]], dtype=torch.long)
    with torch.no_grad():
        logits, weights = model(sample_tokens)
        probs = F.softmax(logits, dim=-1)

    return {
        "tokens": sample_tokens.tolist(),
        "attention_matrix_sample": weights[0].tolist(),
        "probabilities": probs.tolist(),
    }


# ============================================================================
# 7. Using LLM APIs (OpenAI, Anthropic, Ollama, Hugging Face)
# ============================================================================

@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    tokens_used: int
    raw_status: str = "ok"


class UnifiedLLMClient:
    """Unified client for calling LLM APIs with support for:
    - OpenAI format (/v1/chat/completions)
    - Anthropic format (/v1/messages)
    - Ollama local daemon (http://localhost:11434/api/generate)
    - Hugging Face Inference API
    - Built-in Mock / Offline Fallback for automated tests and demonstrations
    """

    def __init__(
        self,
        provider: str = "mock",
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url
        self.model_name = model_name or self._default_model()

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "ollama": "llama3.2:latest",
            "huggingface": "meta-llama/Llama-3.2-1B-Instruct",
            "mock": "deeplearning-lab-assistant",
        }
        return defaults.get(self.provider, "default-model")

    def _call_http(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        """Make a raw JSON POST request with standard library urllib."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> LLMResponse:
        """Generate text across any supported LLM provider."""
        # 1. Mock fallback (works offline without any network/API key requirements)
        if self.provider == "mock" or not self.api_key and self.provider != "ollama":
            response_text = self._mock_generate(prompt)
            return LLMResponse(
                provider=self.provider,
                model=self.model_name,
                content=response_text,
                tokens_used=len(prompt.split()) + len(response_text.split()),
                raw_status="mock_fallback" if self.provider != "mock" else "ok",
            )

        try:
            # 2. OpenAI-compatible API
            if self.provider == "openai":
                url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                body = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                }
                res = self._call_http(url, headers, body)
                content = res["choices"][0]["message"]["content"]
                tokens = res.get("usage", {}).get("total_tokens", 0)
                return LLMResponse(self.provider, self.model_name, content, tokens)

            # 3. Anthropic Messages API
            elif self.provider == "anthropic":
                url = (self.base_url or "https://api.anthropic.com/v1") + "/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                }
                body = {
                    "model": self.model_name,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                }
                res = self._call_http(url, headers, body)
                content = res["content"][0]["text"]
                tokens = res.get("usage", {}).get("output_tokens", 0)
                return LLMResponse(self.provider, self.model_name, content, tokens)

            # 4. Ollama Local REST API
            elif self.provider == "ollama":
                url = (self.base_url or "http://localhost:11434") + "/api/generate"
                headers = {"Content-Type": "application/json"}
                body = {
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
                    "stream": False,
                }
                res = self._call_http(url, headers, body)
                return LLMResponse(self.provider, self.model_name, res.get("response", ""), 0)

            # 5. Hugging Face Inference API
            elif self.provider == "huggingface":
                url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                body = {"inputs": prompt, "parameters": {"max_new_tokens": 128}}
                res = self._call_http(url, headers, body)
                if isinstance(res, list) and len(res) > 0:
                    text = res[0].get("generated_text", str(res[0]))
                else:
                    text = str(res)
                return LLMResponse(self.provider, self.model_name, text, 0)

        except Exception as err:
            fallback = self._mock_generate(prompt)
            return LLMResponse(
                provider=self.provider,
                model=self.model_name,
                content=f"[API Notice: {err} — displaying offline demonstration output]\n\n{fallback}",
                tokens_used=len(prompt.split()),
                raw_status="error_fallback",
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def summarize(self, text: str) -> LLMResponse:
        """Summarize long text using the unified client."""
        prompt = f"Summarize the following deep learning concept into 3 bullet points:\n\n{text}"
        return self.generate(prompt, system_prompt="You are a machine learning technical editor.")

    def answer_question(self, context: str, question: str) -> LLMResponse:
        """Answer a question based on provided technical context."""
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based on the context:"
        return self.generate(prompt, system_prompt="You are a precise technical question-answering assistant.")

    def _mock_generate(self, prompt: str) -> str:
        """Realistic simulated response when working offline or without keys."""
        lower = prompt.lower()
        if "summarize" in lower or "summary" in lower:
            return (
                "• Deep learning leverages multi-layered neural architectures to automatically extract hierarchical representations.\n"
                "• Regularization techniques (Dropout, BatchNorm, Early Stopping) prevent overfitting and stabilize convergence.\n"
                "• Modern workflows combine Transfer Learning and Transformer models to achieve state-of-the-art results efficiently."
            )
        if "question" in lower or "what is" in lower or "how" in lower:
            return (
                "Convolutional neural networks use localized kernel filters with shared weights to capture spatial patterns, "
                "while recurrent architectures (LSTM/GRU) maintain hidden memory states to model sequential and temporal dependencies."
            )
        return (
            f"Demonstration response for: '{prompt[:45]}...'\n\n"
            "Key takeaway: In Week 17–18, we bridge fundamental PyTorch concepts (tensors, autograd, training loops) "
            "with production-grade sequence modeling, transfer learning, and modern LLM orchestration."
        )


# ============================================================================
# Main Demonstration Runner
# ============================================================================

def run_all_demonstrations() -> None:
    """Run all Week 17–18 demonstrations with assertions and readable reports."""
    print("=" * 70)
    print("  WEEK 17–18: DEEP LEARNING, PYTORCH, CNNs, RNNs, TRANSFORMERS & LLMs")
    print("=" * 70)

    # 1. PyTorch Basics
    device = get_device()
    print(f"\n[1] PyTorch Basics & Compute Device: {device}")
    tensor_data = demonstrate_tensors_and_autograd()
    print(f"  • Autograd dy/dx gradient check: {tensor_data['gradients']} (Expected: [14.0, 20.0])")
    _, loss_history = train_linear_model(epochs=80, lr=0.08)
    print(f"  • Linear regression loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f} (decreased!)")

    # 2. CNNs
    print("\n[2] Convolutional Neural Networks (CNNs)")
    cnn_results = test_cnn_architecture()
    print(f"  • Input shape: {cnn_results['input_shape']} -> Output logits: {cnn_results['output_shape']}")
    print(f"  • Sample batch predictions: {cnn_results['predictions']}")

    # 3. RNNs / LSTMs / GRUs
    print("\n[3] Sequence Modeling (RNN vs LSTM vs GRU)")
    seq_results = compare_sequence_models()
    for cell, data in seq_results.items():
        print(f"  • {cell.upper():5}: {data['parameters']:4} parameters | hidden state {data['hidden_shape']}")

    # 4. Regularization
    print("\n[4] Regularization (Dropout, BatchNorm, Early Stopping)")
    reg_results = demonstrate_regularization()
    print(f"  • Early stopping triggered: {reg_results['early_stopped']} at epoch {reg_results['epochs_run']}")
    print(f"  • Best validation loss achieved: {reg_results['best_val_loss']:.4f}")

    # 5. Transfer Learning
    print("\n[5] Transfer Learning (Pretrained CNN Backbone)")
    tl_results = demonstrate_transfer_learning()
    print(f"  • Total parameters: {tl_results['total_params']}")
    print(f"  • Frozen backbone params: {tl_results['frozen_params']} (requires_grad = False)")
    print(f"  • Trainable custom head params: {tl_results['trainable_params']}")

    # 6. Transformers & Attention
    print("\n[6] Hugging Face Transformers & Scaled Dot-Product Attention")
    attn_results = demonstrate_transformer_attention()
    print(f"  • Attention weights row 0 sum: {sum(attn_results['attention_matrix_sample'][0]):.2f} (Softmax normalized)")
    print(f"  • Transformer class probabilities: {[round(p, 3) for p in attn_results['probabilities'][0]]}")

    # 7. LLM APIs (OpenAI, Anthropic, Ollama, Hugging Face)
    print("\n[7] Unified LLM API Client (OpenAI, Anthropic, Ollama, Hugging Face)")
    client = UnifiedLLMClient(provider="mock")
    gen = client.generate("Explain how CNNs preserve spatial features.")
    print(f"  • Generation ({gen.model}): {gen.content[:85]}...")
    summary = client.summarize("Convolutional layers, pooling, backpropagation, and early stopping.")
    print(f"  • Summarization:\n{summary.content}")

    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATIONS AND TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true", help="run tests and exit")
    args = parser.parse_args()
    run_all_demonstrations()


if __name__ == "__main__":
    main()
