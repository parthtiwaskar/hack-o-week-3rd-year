document.addEventListener("DOMContentLoaded", () => {
  async function postJson(url, data) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function renderSvgPath(pathId, values, width = 500, height = 180, pad = 15) {
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
    if (pathElem) pathElem.setAttribute("d", d);
  }

  // 1. Status Check
  fetch("/api/status")
    .then((r) => r.json())
    .then((s) => {
      const el = document.getElementById("lib-status-text");
      if (el) el.textContent = `XGBoost ${s.xgboost_version} · LightGBM ${s.lightgbm_version}`;
    })
    .catch(() => {});

  // 2. Ensemble Benchmark
  const samplesSlider = document.getElementById("ens-samples");
  const samplesDisplay = document.getElementById("samples-display");
  const noiseSlider = document.getElementById("ens-noise");
  const noiseDisplay = document.getElementById("noise-display");
  const btnBenchmark = document.getElementById("btn-benchmark-ensembles");

  if (samplesSlider && samplesDisplay) {
    samplesSlider.addEventListener("input", (e) => {
      samplesDisplay.textContent = e.target.value;
    });
  }
  if (noiseSlider && noiseDisplay) {
    noiseSlider.addEventListener("input", (e) => {
      noiseDisplay.textContent = `${e.target.value}%`;
    });
  }

  if (btnBenchmark) {
    btnBenchmark.addEventListener("click", async () => {
      btnBenchmark.disabled = true;
      btnBenchmark.textContent = "Training Ensembles…";

      try {
        const res = await postJson("/api/ensemble/benchmark", {
          n_samples: parseInt(samplesSlider.value, 10),
          noise_ratio: parseFloat(noiseSlider.value) / 100.0,
        });

        const tbody = document.getElementById("ensemble-table-body");
        if (tbody) {
          const rows = [
            { name: "Single Decision Tree", type: "Baseline" },
            { name: "Bagging (Random Forest)", type: "Bagging" },
            { name: "AdaBoost", type: "Boosting" },
            { name: "Gradient Boosting (sklearn)", type: "Boosting" },
            { name: "XGBoost", type: "Optimized Boosting", highlight: true },
            { name: "LightGBM", type: "Leaf-wise Boosting", highlight: true },
          ];

          tbody.innerHTML = rows
            .map((r) => {
              const m = res.models[r.name] || {};
              const testAcc = m.test_accuracy ? `${(m.test_accuracy * 100).toFixed(1)}%` : "—";
              const testAuc = m.test_auc ? m.test_auc.toFixed(3) : "—";
              const trainAcc = m.train_accuracy ? `${(m.train_accuracy * 100).toFixed(1)}%` : "—";
              const cls = r.highlight ? "class='highlight-row'" : "";
              return `<tr ${cls}><td>${r.name}</td><td>${r.type}</td><td><strong>${testAcc}</strong></td><td>${testAuc}</td><td>${trainAcc}</td></tr>`;
            })
            .join("");
        }
      } catch (err) {
        alert("Benchmark error: " + err.message);
      } finally {
        btnBenchmark.disabled = false;
        btnBenchmark.textContent = "Train & Benchmark Ensembles";
      }
    });
  }

  // 3. Bias-Variance Decomposition
  const btnBiasVariance = document.getElementById("btn-run-biasvariance");
  if (btnBiasVariance) {
    btnBiasVariance.addEventListener("click", async () => {
      btnBiasVariance.disabled = true;
      btnBiasVariance.textContent = "Computing Decomposition…";
      const empty = document.getElementById("bv-chart-empty");
      if (empty) empty.style.display = "none";

      try {
        const res = await postJson("/api/bias-variance/decompose", {});
        const depths = res.depths;
        const biases = depths.map((d) => res.decomposition[d].bias_squared);
        const vars = depths.map((d) => res.decomposition[d].variance);
        const totals = depths.map((d) => res.decomposition[d].total_expected_mse);

        renderSvgPath("bv-bias-path", biases);
        renderSvgPath("bv-var-path", vars);
        renderSvgPath("bv-total-path", totals);

        const pill = document.getElementById("bv-sweet-spot");
        if (pill) {
          pill.textContent = `Optimal Sweet Spot: Depth = ${res.sweet_spot_depth} (Min Total MSE)`;
        }
        const text = document.getElementById("bv-summary-text");
        if (text) {
          text.innerHTML = `At Depth 1: Bias² is high (underfitting). At Depth 14: Variance dominates (overfitting).<br>Optimal depth <strong>${res.sweet_spot_depth}</strong> balances the trade-off.`;
        }
      } catch (err) {
        alert("Bias-variance error: " + err.message);
      } finally {
        btnBiasVariance.disabled = false;
        btnBiasVariance.textContent = "Compute Bias-Variance Decomposition";
      }
    });
  }

  // 4. Overfitting Curves
  const btnOverfitting = document.getElementById("btn-run-overfitting");
  if (btnOverfitting) {
    btnOverfitting.addEventListener("click", async () => {
      btnOverfitting.disabled = true;
      btnOverfitting.textContent = "Plotting Capacity…";
      const empty = document.getElementById("overfit-empty");
      if (empty) empty.style.display = "none";

      try {
        const res = await postJson("/api/overfitting/curve", {});
        renderSvgPath("overfit-train-path", res.train_accuracies);
        renderSvgPath("overfit-val-path", res.validation_accuracies);

        const badge = document.getElementById("overfit-gap-badge");
        if (badge) {
          badge.textContent = `Generalization Gap at Depth 15: ${(res.deep_overfit_gap * 100).toFixed(1)}% (Overfitting)`;
        }
      } catch (err) {
        alert("Overfitting test error: " + err.message);
      } finally {
        btnOverfitting.disabled = false;
        btnOverfitting.textContent = "Plot Capacity Curves";
      }
    });
  }

  // 5. Regularization Evaluation
  const btnReg = document.getElementById("btn-run-regularization");
  if (btnReg) {
    btnReg.addEventListener("click", async () => {
      btnReg.disabled = true;
      btnReg.textContent = "Evaluating Penalties…";

      try {
        const res = await postJson("/api/regularization/compare", {});
        const l1 = res.l1_sparsity;
        const xgbReg = res.xgboost_regularization;

        document.getElementById("l1-zeroed").textContent = `${l1.lasso_zeroed_features} / ${l1.noise_features_count}`;
        document.getElementById("xgb-reg-imp").textContent = `+${xgbReg.improvement}%`;
      } catch (err) {
        alert("Regularization error: " + err.message);
      } finally {
        btnReg.disabled = false;
        btnReg.textContent = "Evaluate L1/L2 on 15 Noisy Features";
      }
    });
  }

  // Initial trigger for preview
  if (btnBenchmark) btnBenchmark.click();
});
