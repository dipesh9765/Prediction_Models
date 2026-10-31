"use strict";

// Builds the whole UI from GET /v1/models and GET /v1/models/{key}. Nothing about a model is
// hard-coded here. All dynamic text goes through textContent, never innerHTML.

const tabs = document.getElementById("tabs");
const panel = document.getElementById("panel");

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "text") node.textContent = value;
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ")
      : body.detail || response.statusText;
    throw new Error(detail);
  }
  return body;
}

function numberInput(feature) {
  const input = el("input", { type: "number", name: feature.name, required: !feature.optional });
  input.min = feature.min;
  input.max = feature.max;
  input.step = feature.integer ? "1" : "any";
  // Optional fields start empty (the model imputes them); everything else starts with a valid example.
  if (feature.example !== null && !feature.optional) input.value = String(feature.example);
  else if (feature.example !== null) input.placeholder = `e.g. ${feature.example}`;
  return input;
}

function selectInput(feature) {
  const select = el("select", { name: feature.name, required: true });
  for (const choice of feature.choices) select.append(el("option", { value: choice, text: choice }));
  if (feature.example) select.value = feature.example;
  return select;
}

function field(feature) {
  const unit = feature.unit ? ` (${feature.unit})` : "";
  const hint = feature.optional ? " - optional" : "";
  const input = feature.kind === "category" ? selectInput(feature) : numberInput(feature);
  return el("label", {}, el("span", { text: feature.label + unit + hint }), input);
}

function readPayload(form, features) {
  const payload = {};
  for (const feature of features) {
    const raw = form.elements[feature.name].value;
    if (feature.kind === "category") payload[feature.name] = raw;
    else if (raw === "") payload[feature.name] = feature.optional ? null : raw;
    else payload[feature.name] = Number(raw);
  }
  return payload;
}

function format(value, unit) {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
  return unit ? `${Number(rounded).toLocaleString()} ${unit}` : rounded;
}

function renderResult(target, result, model) {
  target.replaceChildren();
  if (result.probability !== null) {
    const pct = Math.round(result.probability * 100);
    target.append(
      el("div", { className: "value", text: `${pct}%` }),
      el("div", { text: result.label }),
      el("div", { className: "meter" }, el("span", { style: `width:${pct}%` })),
    );
  } else {
    target.append(el("div", { className: "value", text: format(result.prediction, result.unit) }));
    if (result.lower !== null) {
      const level = Math.round(result.interval_level * 100);
      target.append(el("div", { className: "range",
        text: `${level}% interval: ${format(result.lower, "")} to ${format(result.upper, result.unit)}` }));
    }
  }
  for (const warning of result.warnings) target.append(el("div", { className: "banner warn", text: warning }));
  target.append(el("div", { className: "meta",
    text: `Model ${model.selected_model}, version ${result.version}` }));
}

function renderCard(model) {
  const card = model.card;
  if (!card) return el("span");
  const rows = Object.entries(card.metrics.test).map(([name, value]) =>
    el("tr", {}, el("th", { text: name }), el("td", { text: String(value) }),
      el("td", { text: String(card.metrics.baseline[name]) })));
  const table = el("table", {}, el("tr", {}, el("th", { text: "Test metric" }),
    el("th", { text: "Model" }), el("th", { text: "Naive baseline" })), ...rows);
  const limits = el("ul", {}, ...card.limitations.map((text) => el("li", { text })));
  return el("details", {}, el("summary", { text: "How good is this model?" }), table,
    el("p", { text: `Trained on ${card.data.train_rows.toLocaleString()} rows; evaluated on `
      + `${card.data.test_rows.toLocaleString()} held-out rows.` }), limits);
}

async function showModel(summary) {
  panel.replaceChildren(el("p", { text: "Loading..." }));
  const model = await getJson(`/v1/models/${summary.key}`);
  const result = el("div", { id: "result" });
  const submit = el("button", { type: "submit", className: "submit", text: "Predict" });
  const form = el("form", {}, ...model.features.map(field), el("div", { className: "actions" }, submit));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submit.disabled = true;
    result.replaceChildren();
    try {
      const answer = await getJson(`/v1/${model.key}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readPayload(form, model.features)),
      });
      renderResult(result, answer, model);
    } catch (error) {
      result.replaceChildren(el("div", { className: "banner bad", text: error.message }));
    } finally {
      submit.disabled = false;
    }
  });

  panel.replaceChildren(el("h2", { text: model.title }), el("p", { className: "lead", text: model.description }));
  if (model.disclaimer) panel.append(el("div", { className: "banner warn", text: model.disclaimer }));
  if (!model.available) {
    panel.append(el("div", { className: "banner bad", text: `Model unavailable: ${model.reason}` }));
    return;
  }
  panel.append(form, result, renderCard(model));
}

async function main() {
  try {
    const models = await getJson("/v1/models");
    const buttons = models.map((model) => {
      const button = el("button", { type: "button", text: model.title, role: "tab",
        disabled: !model.available });
      button.setAttribute("aria-selected", "false");
      button.addEventListener("click", () => {
        buttons.forEach((b) => b.setAttribute("aria-selected", String(b === button)));
        showModel(model).catch((e) => panel.replaceChildren(el("div", { className: "banner bad", text: e.message })));
      });
      return button;
    });
    tabs.replaceChildren(...buttons);
    const first = buttons.find((b) => !b.disabled);
    if (first) first.click();
    else panel.replaceChildren(el("div", { className: "banner bad", text: "No model is available." }));
  } catch (error) {
    panel.replaceChildren(el("div", { className: "banner bad", text: error.message }));
  }
}

main();
