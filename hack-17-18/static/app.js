document.addEventListener("DOMContentLoaded", () => {
  // Helper for JSON POST requests
  async function postJson(endpoint, data) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP error ${response.status}`);
    }
    return response.json();
  }

  // Draw smooth SVG path from numerical loss array
  function renderSvgPath(svgId, pathId, values, width = 500, height = 180, pad = 15) {
    if (!values || values.length === 0) return;
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = maxVal - minVal || 1.0;

    const points = values.map((val, idx) => {
      const x = pad + (idx / (values.length - 1 || 1)) * (width - 2 * pad);
      const normalized = (val - minVal) / range;
      const y = height - pad - normalized * (height - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const d = "M " + points.join(" L ");
    const pathElem = document.getElementById(pathId);
    if (pathElem) {
      pathElem.setAttribute("d", d);
    }
  }

  // =========================================================================
  // Status & Device Detection
  // =========================================================================
  fetch("/api/status")
    .then((r) => r.json())
    .then((status) => {
      const deviceText = document.getElementById("device-text");
      if (deviceText) {
        deviceText.textContent = `PyTorch v${status.torch_version} · Device: ${status.device.toUpperCase()}`;
      }
    })
    .catch(() => {
      const deviceText = document.getElementById("device-text");
      if (deviceText) deviceText.textContent = "Lab API Connected";
    });

  // =========================================================================
  // Module 01: PyTorch & Autograd Training Loop
  // =========================================================================
  const epochsSlider = document.getElementById("pytorch-epochs");
  const epochsDisplay = document.getElementById("epochs-display");
  const lrSlider = document.getElementById("pytorch-lr");
  const lrDisplay = document.getElementById("lr-display");
  const btnTrainPytorch = document.getElementById("btn-train-pytorch");

  if (epochsSlider && epochsDisplay) {
    epochsSlider.addEventListener("input", (e) => {
      epochsDisplay.textContent = e.target.value;
    });
  }
  if (lrSlider && lrDisplay) {
    lrSlider.addEventListener("input", (e) => {
      lrDisplay.textContent = e.target.value;
    });
  }

  if (btnTrainPytorch) {
    btnTrainPytorch.addEventListener("click", async () => {
      btnTrainPytorch.disabled = true;
      btnTrainPytorch.textContent = "Training with Adam…";
      const emptyText = document.getElementById("pytorch-loss-empty");
      if (emptyText) emptyText.style.display = "none";

      try {
        const res = await postJson("/api/train/pytorch", {
          model_type: document.getElementById("pytorch-model").value,
          epochs: parseInt(epochsSlider.value, 10),
          lr: parseFloat(lrSlider.value),
        });

        renderSvgPath("pytorch-loss-chart", "pytorch-loss-path", res.loss_history);

        const statsBar = document.getElementById("pytorch-stats");
        if (statsBar) {
          statsBar.innerHTML = `
            <span>Initial: ${res.initial_loss.toFixed(4)}</span>
            <span>Final: <strong>${res.final_loss.toFixed(4)}</strong></span>
            <span>Convergence: ${(
              ((res.initial_loss - res.final_loss) / res.initial_loss) *
              100
            ).toFixed(1)}%</span>
          `;
        }

        const autogradDisplay = document.getElementById("autograd-display");
        if (autogradDisplay) {
          autogradDisplay.textContent = `// Autograd Verified dynamically
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = 3*(x**2) + 2*x + 1
y.sum().backward()

Gradients computed: ${JSON.stringify(res.autograd_gradients)}
Calculated Loss: ${res.autograd_loss.toFixed(2)}
Model parameters updated via Adam step!`;
        }
      } catch (err) {
        alert("Training error: " + err.message);
      } finally {
        btnTrainPytorch.disabled = false;
        btnTrainPytorch.textContent = "Run Training Loop";
      }
    });
  }

  // =========================================================================
  // Module 02: CNN & Feature Maps
  // =========================================================================
  const btnRunCnn = document.getElementById("btn-run-cnn");
  if (btnRunCnn) {
    btnRunCnn.addEventListener("click", async () => {
      btnRunCnn.disabled = true;
      btnRunCnn.textContent = "Running Convolutions…";

      try {
        const res = await postJson("/api/cnn/infer", {
          pattern_type: document.getElementById("cnn-pattern").value,
        });

        document.getElementById("cnn-block1-dim").textContent = JSON.stringify(res.block1_shape);
        document.getElementById("cnn-block2-dim").textContent = JSON.stringify(res.block2_shape);

        const barsContainer = document.getElementById("cnn-prob-bars");
        if (barsContainer) {
          barsContainer.innerHTML = res.classes
            .map((cls, idx) => {
              const prob = res.probabilities[idx];
              const pct = (prob * 100).toFixed(1);
              return `
              <div class="prob-row">
                <span>${cls}:</span>
                <div class="bar-fill" style="width: ${pct}%"></div>
                <span class="prob-num">${prob.toFixed(2)}</span>
              </div>
            `;
            })
            .join("");
        }
      } catch (err) {
        alert("CNN error: " + err.message);
      } finally {
        btnRunCnn.disabled = false;
        btnRunCnn.textContent = "Pass Through ConvNet";
      }
    });
  }

  // =========================================================================
  // Module 03: Sequence Modeling (RNN vs LSTM vs GRU)
  // =========================================================================
  const seqSlider = document.getElementById("seq-length");
  const seqDisplay = document.getElementById("seq-len-display");
  const btnRunSeq = document.getElementById("btn-run-seq");

  if (seqSlider && seqDisplay) {
    seqSlider.addEventListener("input", (e) => {
      seqDisplay.textContent = e.target.value;
    });
  }

  if (btnRunSeq) {
    btnRunSeq.addEventListener("click", async () => {
      btnRunSeq.disabled = true;
      btnRunSeq.textContent = "Benchmarking Sequences…";

      try {
        const res = await postJson("/api/sequence/compare", {
          seq_len: parseInt(seqSlider.value, 10),
        });

        document.getElementById("rnn-params").textContent =
          res.models.rnn.parameters.toLocaleString();
        document.getElementById("lstm-params").textContent =
          res.models.lstm.parameters.toLocaleString();
        document.getElementById("gru-params").textContent =
          res.models.gru.parameters.toLocaleString();
      } catch (err) {
        alert("Sequence error: " + err.message);
      } finally {
        btnRunSeq.disabled = false;
        btnRunSeq.textContent = "Compare Architectures";
      }
    });
  }

  // =========================================================================
  // Module 04: Regularization & Overfitting
  // =========================================================================
  const btnRunReg = document.getElementById("btn-run-reg");
  if (btnRunReg) {
    btnRunReg.addEventListener("click", async () => {
      btnRunReg.disabled = true;
      btnRunReg.textContent = "Evaluating Generalization…";
      const emptyText = document.getElementById("reg-chart-empty");
      if (emptyText) emptyText.style.display = "none";

      try {
        const res = await postJson("/api/regularization/test", {
          use_dropout: document.getElementById("reg-dropout").checked,
          use_batchnorm: document.getElementById("reg-batchnorm").checked,
          use_earlystop: document.getElementById("reg-earlystop").checked,
        });

        renderSvgPath("reg-loss-chart", "reg-train-path", res.train_losses);
        renderSvgPath("reg-loss-chart", "reg-val-path", res.val_losses);

        const pill = document.getElementById("reg-status-pill");
        if (pill) {
          if (res.early_stopped) {
            pill.textContent = `Early Stopping triggered at epoch ${res.epochs_completed}! Restored best weights.`;
            pill.style.backgroundColor = "#273822";
            pill.style.color = "#34d399";
          } else {
            pill.textContent = `Completed all ${res.epochs_completed} epochs without divergence.`;
            pill.style.backgroundColor = "#172433";
            pill.style.color = "#38bdf8";
          }
        }
      } catch (err) {
        alert("Regularization test error: " + err.message);
      } finally {
        btnRunReg.disabled = false;
        btnRunReg.textContent = "Train & Test Overfitting";
      }
    });
  }

  // =========================================================================
  // Module 05: Transfer Learning
  // =========================================================================
  const classesSlider = document.getElementById("transfer-classes");
  const classesDisplay = document.getElementById("classes-display");
  const btnRunTransfer = document.getElementById("btn-run-transfer");

  if (classesSlider && classesDisplay) {
    classesSlider.addEventListener("input", (e) => {
      classesDisplay.textContent = e.target.value;
    });
  }

  if (btnRunTransfer) {
    btnRunTransfer.addEventListener("click", async () => {
      btnRunTransfer.disabled = true;
      try {
        const res = await postJson("/api/transfer/setup", {
          num_classes: parseInt(classesSlider.value, 10),
        });

        document.getElementById("tl-total-params").textContent =
          res.total_parameters.toLocaleString();
        document.getElementById(
          "tl-frozen-params"
        ).textContent = `${res.frozen_parameters.toLocaleString()} (${res.frozen_percentage}%)`;
        document.getElementById("tl-trainable-params").textContent =
          res.trainable_parameters.toLocaleString();
      } catch (err) {
        alert("Transfer learning error: " + err.message);
      } finally {
        btnRunTransfer.disabled = false;
      }
    });
  }

  // =========================================================================
  // Module 06: Transformers & Self-Attention Matrix
  // =========================================================================
  const btnRunAttention = document.getElementById("btn-run-attention");
  if (btnRunAttention) {
    btnRunAttention.addEventListener("click", async () => {
      const textInput = document.getElementById("transformer-input").value;
      btnRunAttention.disabled = true;
      btnRunAttention.textContent = "Computing Attention…";

      try {
        const res = await postJson("/api/transformer/attention", { text: textInput });
        const grid = document.getElementById("attention-matrix-grid");
        if (grid) {
          grid.innerHTML = "";
          const n = res.tokens.length;
          grid.style.gridTemplateColumns = `repeat(${n}, 44px)`;

          res.attention_matrix.forEach((row, rIdx) => {
            row.forEach((val, cIdx) => {
              const cell = document.createElement("div");
              cell.className = "attn-cell";
              // Color map based on attention strength
              const alpha = Math.max(0.15, Math.min(1.0, val * 2.5));
              cell.style.backgroundColor = `rgba(56, 189, 248, ${alpha})`;
              cell.title = `Attn(${res.tokens[rIdx]} → ${res.tokens[cIdx]}): ${val}`;
              cell.textContent = val.toFixed(2);
              grid.appendChild(cell);
            });
          });
        }

        const predElem = document.getElementById("transformer-prediction");
        if (predElem) {
          predElem.textContent = `${res.prediction} (P=${res.sentiment_probabilities[1].toFixed(2)})`;
        }
      } catch (err) {
        alert("Attention computation error: " + err.message);
      } finally {
        btnRunAttention.disabled = false;
        btnRunAttention.textContent = "Compute Attention Heatmap";
      }
    });
  }

  // =========================================================================
  // Module 07: Unified LLM APIs
  // =========================================================================
  const btnRunLlm = document.getElementById("btn-run-llm");
  if (btnRunLlm) {
    btnRunLlm.addEventListener("click", async () => {
      btnRunLlm.disabled = true;
      btnRunLlm.textContent = "Querying Model…";
      const contentElem = document.getElementById("llm-response-content");
      const metaElem = document.getElementById("llm-meta");
      contentElem.textContent = "Connecting to LLM backend…";

      try {
        const res = await postJson("/api/llm/query", {
          provider: document.getElementById("llm-provider").value,
          task: document.getElementById("llm-task").value,
          prompt: document.getElementById("llm-prompt").value,
          api_key: document.getElementById("llm-apikey").value || null,
        });

        contentElem.textContent = res.content;
        metaElem.textContent = `Provider: ${res.provider.toUpperCase()} · Model: ${res.model} · Tokens: ${res.tokens_used}`;
      } catch (err) {
        contentElem.textContent = `Error: ${err.message}`;
      } finally {
        btnRunLlm.disabled = false;
        btnRunLlm.textContent = "Dispatch LLM Query";
      }
    });
  }

  // Initial trigger for preview
  if (btnRunAttention) btnRunAttention.click();
});
