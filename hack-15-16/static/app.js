const optimizerButtons = [...document.querySelectorAll("[data-optimizer]")];
const epochsSlider = document.querySelector("#epochs");
const trainButton = document.querySelector("#train-button");
const trainMessage = document.querySelector("#training-message");
const chartLine = document.querySelector("#chart-line");
const chartArea = document.querySelector("#chart-area");
const chartEndpoint = document.querySelector("#chart-endpoint");
const chartEmpty = document.querySelector("#chart-empty");
const chartStatus = document.querySelector("#chart-status");
const predictButton = document.querySelector("#predict-button");
let selectedOptimizer = "adam";

function setMessage(message, isError = false) {
  trainMessage.textContent = message;
  trainMessage.classList.toggle("error", isError);
}

function formatLoss(value) {
  return Number(value).toFixed(4);
}

function drawChart(losses) {
  const width = 560;
  const top = 10;
  const bottom = 202;
  const chartHeight = bottom - top;
  const xStep = losses.length > 1 ? width / (losses.length - 1) : 0;
  const points = losses.map((loss, index) => {
    const x = index * xStep;
    const y = bottom - Math.min(1, Math.max(0, loss)) * chartHeight;
    return [x, y];
  });
  const line = points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x},${y}`).join(" ");
  chartLine.setAttribute("d", line);
  chartArea.setAttribute("d", `${line} L${width},${bottom} L0,${bottom} Z`);
  chartEndpoint.setAttribute("cx", String(points.at(-1)[0]));
  chartEndpoint.setAttribute("cy", String(points.at(-1)[1]));
  chartEmpty.hidden = true;
  chartStatus.textContent = "TRAINING COMPLETE";
  chartStatus.classList.add("active");
  document.querySelector("#loss-readout strong").textContent = formatLoss(losses.at(-1));
}

function updateTruthTable(rows) {
  const tableRows = document.querySelectorAll("#truth-table-body tr");
  rows.forEach((row, index) => {
    const cell = tableRows[index].querySelector(".network-cell");
    const matches = row.prediction === row.target;
    cell.textContent = String(row.prediction);
    cell.classList.toggle("pending", false);
    cell.classList.toggle("match", matches);
    cell.classList.toggle("mismatch", !matches);
  });
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

optimizerButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedOptimizer = button.dataset.optimizer;
    optimizerButtons.forEach((option) => {
      const selected = option === button;
      option.classList.toggle("selected", selected);
      option.setAttribute("aria-pressed", String(selected));
    });
  });
});

epochsSlider.addEventListener("input", () => {
  const progress = ((Number(epochsSlider.value) - 100) / 7900) * 100;
  epochsSlider.style.background = `linear-gradient(90deg, var(--mint) ${progress}%, #2a3938 ${progress}%)`;
  document.querySelector("#epochs-value").textContent = Number(epochsSlider.value).toLocaleString();
});

trainButton.addEventListener("click", async () => {
  trainButton.disabled = true;
  predictButton.disabled = true;
  trainButton.querySelector("span:nth-child(2)").textContent = "Learning in progress…";
  setMessage("Forward pass. Backward pass. Adjusting weights…");
  chartStatus.textContent = "TRAINING";
  chartStatus.classList.remove("active");

  try {
    const result = await postJson("/api/train", {
      optimizer: selectedOptimizer,
      epochs: Number(epochsSlider.value),
    });
    drawChart(result.loss_history);
    updateTruthTable(result.predictions);
    document.querySelector("#accuracy-value").innerHTML = `${Math.round(result.accuracy * 100)}<small>%</small>`;
    document.querySelector("#starting-loss").textContent = formatLoss(result.initial_loss);
    document.querySelector("#ending-loss").textContent = formatLoss(result.final_loss);
    document.querySelector("#result-title").textContent = "Your network learned XOR";
    document.querySelector("#result-caption").textContent =
      `${result.optimizer.toUpperCase()} · ${result.epochs.toLocaleString()} epochs · predictions match the truth table.`;
    setMessage("Training complete. The loss came down as the network learned.");
    predictButton.disabled = false;
    await predictInput();
  } catch (error) {
    setMessage(error.message, true);
    chartStatus.textContent = "TRAINING FAILED";
  } finally {
    trainButton.disabled = false;
    trainButton.querySelector("span:nth-child(2)").textContent = "Train my network";
  }
});

async function predictInput() {
  const first = document.querySelector("#input-first").value;
  const second = document.querySelector("#input-second").value;
  const result = await postJson("/api/predict", { first, second });
  const row = result.inputs.join(" · ");
  const mark = document.querySelector(".prediction-mark");
  mark.textContent = String(result.prediction);
  mark.classList.toggle("correct", result.prediction === (result.inputs[0] ^ result.inputs[1]));
  mark.classList.toggle("incorrect", result.prediction !== (result.inputs[0] ^ result.inputs[1]));
  document.querySelector(".prediction-copy strong").textContent =
    `Inputs ${row} → output ${result.prediction}`;
  document.querySelector(".prediction-copy span").textContent =
    `The network is ${(result.probability * 100).toFixed(1)}% confident the answer is 1.`;
  document.querySelector(".confidence strong").textContent =
    `${(Math.max(result.probability, 1 - result.probability) * 100).toFixed(1)}%`;
}

predictButton.addEventListener("click", async () => {
  predictButton.disabled = true;
  try {
    await predictInput();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    predictButton.disabled = false;
  }
});

document.querySelectorAll("#input-first, #input-second").forEach((input) => {
  input.addEventListener("change", () => {
    if (!predictButton.disabled) predictButton.click();
  });
});

fetch("/api/status")
  .then((response) => {
    if (!response.ok) throw new Error("Could not connect to the training server.");
    return response.json();
  })
  .then((status) => {
    if (!status.trained) return;
    predictButton.disabled = false;
  })
  .catch((error) => setMessage(error.message, true));
