"use strict";

const $ = (selector) => document.querySelector(selector);
const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 0 });
const decimal = (digits) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style") Object.assign(node.style, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Falha ao carregar ${url}`);
  return response.json();
}

/* ---------- Navegação entre vistas ---------- */

function showView() {
  const name = location.hash.startsWith("#projeto") || location.hash.startsWith("#sec-") ? "projeto" : "analisar";
  for (const view of document.querySelectorAll(".view")) view.hidden = view.dataset.view !== name;
  for (const link of document.querySelectorAll(".nav a")) {
    if (link.dataset.view === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (name === "projeto") loadProject();
}
window.addEventListener("hashchange", showView);

/* ---------- Estado do modelo ---------- */

function renderStatus(model) {
  if (!model) return;
  const status = $("#model-status");
  status.dataset.state = model.state;
  const device = String(model.device).toLowerCase() === "cpu" ? "CPU" : `GPU ${model.device}`;
  const text = {
    missing: "Pesos ausentes",
    idle: `Modelo pronto em ${device}`,
    loaded: `Modelo carregado em ${device}`,
  }[model.state] || model.message;
  status.querySelector(".status-text").textContent = text;
  status.title = model.message;
  const parts = [`${model.weights}, entrada de ${model.imgsz} px`];
  if (model.sha256) parts.push(`SHA-256 ${model.sha256}`);
  $("#model-line").textContent = model.state === "missing" ? model.message : parts.join(", ");
}

/* ---------- Análise ---------- */

const state = {
  mode: "image",
  files: { image: null, video: null },
  urls: { image: null, video: null },
  names: { image: "", video: "" },
  results: { image: null, video: null },
  busy: false,
  defaultConfidence: 0.35,
};

const confidence = $("#confidence");
const formatConfidence = (value) => decimal(2).format(Number(value));

function setConfidenceLabel() {
  $("#confidence-value").textContent = formatConfidence(confidence.value);
  const result = state.results[state.mode];
  $("#stale").hidden = !result || Math.abs(result.confidence - Number(confidence.value)) < 1e-6;
}
confidence.addEventListener("input", setConfidenceLabel);

function setMode(mode) {
  state.mode = mode;
  $("#tab-image").setAttribute("aria-selected", String(mode === "image"));
  $("#tab-video").setAttribute("aria-selected", String(mode === "video"));
  $("#analyze-image").hidden = mode !== "image";
  $("#analyze-video").hidden = mode !== "video";
  $("#dropzone").setAttribute("for", mode === "image" ? "file-image" : "file-video");
  $("#drop-title").textContent = mode === "image" ? "Arraste uma imagem para cá" : "Arraste um vídeo para cá";
  $("#drop-sub").textContent = mode === "image"
    ? "JPG, PNG ou WebP com até 20 MB"
    : "MP4 com até 30 segundos e 100 MB";
  hideAlert();
  renderStage();
  renderResult();
  setConfidenceLabel();
}
$("#tab-image").addEventListener("click", () => setMode("image"));
$("#tab-video").addEventListener("click", () => setMode("video"));
document.querySelector(".segmented").addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const next = state.mode === "image" ? "video" : "image";
  setMode(next);
  $(next === "image" ? "#tab-image" : "#tab-video").focus();
});

function setFile(kind, file, name = file?.name || "") {
  if (state.urls[kind]) URL.revokeObjectURL(state.urls[kind]);
  state.files[kind] = file;
  state.names[kind] = name;
  state.urls[kind] = file ? URL.createObjectURL(file) : null;
  state.results[kind] = null;
  for (const sample of document.querySelectorAll(".sample")) {
    sample.setAttribute("aria-pressed", String(sample.dataset.name === name && sample.dataset.kind === kind));
  }
  if (state.mode !== kind) setMode(kind);
  hideAlert();
  renderStage();
  renderResult();
}

function acceptFile(file) {
  if (!file) return;
  const kind = file.type.startsWith("video/") ? "video" : file.type.startsWith("image/") ? "image" : null;
  if (!kind) return showAlert("Formato não suportado. Envie uma imagem JPG, PNG ou WebP, ou um vídeo MP4.");
  setFile(kind, file);
}

$("#file-image").addEventListener("change", (event) => { acceptFile(event.target.files[0]); event.target.value = ""; });
$("#file-video").addEventListener("change", (event) => { acceptFile(event.target.files[0]); event.target.value = ""; });
$("#replace-file").addEventListener("click", () => $(state.mode === "image" ? "#file-image" : "#file-video").click());

const stage = $("#stage");
for (const type of ["dragenter", "dragover"]) {
  stage.addEventListener(type, (event) => { event.preventDefault(); stage.classList.add("is-dragging"); });
}
for (const type of ["dragleave", "drop"]) {
  stage.addEventListener(type, (event) => { event.preventDefault(); stage.classList.remove("is-dragging"); });
}
stage.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));

function renderStage() {
  const kind = state.mode;
  const url = state.urls[kind];
  const result = state.results[kind];
  stage.dataset.state = state.busy ? "busy" : !url ? "empty" : result ? "done" : "ready";
  $("#viewer").hidden = !url;
  $("#analyze-image").disabled = !state.files.image || state.busy;
  $("#analyze-video").disabled = !state.files.video || state.busy;
  if (!url) return;
  $("#file-name").textContent = state.names[kind];
  const isImage = kind === "image";
  $("#frame").hidden = !isImage;
  $("#toggle-boxes").hidden = !(isImage && result && result.count);
  const video = $("#video-preview");
  video.hidden = isImage;
  if (isImage) {
    const preview = $("#preview");
    if (preview.getAttribute("src") !== url) preview.src = url;
    preview.alt = result ? `${state.names.image} com ${result.count} regiões marcadas` : state.names.image;
    drawBoxes(result);
  } else {
    const source = result && result.playable ? result.files.video : url;
    if (video.getAttribute("src") !== source) video.src = source;
  }
}

function drawBoxes(result) {
  const layer = $("#boxes");
  layer.replaceChildren();
  layer.classList.remove("has-focus");
  if (!result) return;
  result.detections.forEach((item, index) => {
    const [x1, y1, x2, y2] = item.bbox;
    const box = el("div", {
      class: "box" + (y1 / result.height < 0.06 ? " is-bottom" : ""),
      "data-index": index,
      style: {
        left: `${(x1 / result.width) * 100}%`, top: `${(y1 / result.height) * 100}%`,
        width: `${((x2 - x1) / result.width) * 100}%`, height: `${((y2 - y1) / result.height) * 100}%`,
      },
    }, [el("span", { class: "box-tag" }, [el("b", { text: index + 1 }), percent.format(item.confidence)])]);
    layer.append(box);
  });
}

function focusBox(index) {
  const layer = $("#boxes");
  layer.classList.toggle("has-focus", index !== null);
  for (const box of layer.children) box.classList.toggle("is-active", Number(box.dataset.index) === index);
  for (const row of document.querySelectorAll(".detection")) row.classList.toggle("is-active", Number(row.dataset.index) === index);
}

$("#toggle-boxes").addEventListener("click", (event) => {
  const visible = event.currentTarget.getAttribute("aria-pressed") !== "true";
  event.currentTarget.setAttribute("aria-pressed", String(visible));
  event.currentTarget.textContent = visible ? "Caixas visíveis" : "Caixas ocultas";
  $("#boxes").classList.toggle("is-hidden", !visible);
});

function renderResult() {
  const result = state.results[state.mode];
  $("#result").dataset.state = result ? "done" : "idle";
  $("#result-empty").hidden = Boolean(result);
  $("#result-image").hidden = !(result && result.kind === "image");
  $("#result-video").hidden = !(result && result.kind === "video");
  $("#downloads").hidden = !result;
  setConfidenceLabel();
  if (!result) return;
  const row = $("#download-row");
  row.replaceChildren();
  const labels = result.kind === "image"
    ? [["image", "Imagem anotada"], ["json", "JSON"], ["csv", "CSV"]]
    : [["video", "Vídeo anotado"], ["json", "JSON"], ["csv", "CSV"]];
  for (const [key, label] of labels) {
    row.append(el("a", { class: "button button-line", href: result.files[key], download: "", text: label }));
  }
  if (result.kind === "image") renderImageResult(result);
  else renderVideoResult(result);
}

function renderImageResult(result) {
  $("#count").textContent = result.count;
  $("#count-label").textContent = result.count === 1 ? "resíduo detectado" : "resíduos detectados";
  const time = result.duration_ms === null ? "" : ` em ${decimal(0).format(result.duration_ms)} ms`;
  $("#image-meta").textContent = `Processamento${time}, acima de ${formatConfidence(result.confidence)} de confiança. Imagem de ${result.width} × ${result.height} px.`;
  $("#empty-note").hidden = result.count > 0;
  const list = $("#detections");
  list.replaceChildren();
  result.detections.forEach((item, index) => {
    const [x1, y1, x2, y2] = item.bbox.map((value) => Math.round(value));
    const row = el("li", {
      class: "detection", tabindex: "0", "data-index": index,
      "aria-label": `Região ${index + 1}, resíduo com ${percent.format(item.confidence)} de confiança`,
      onmouseenter: () => focusBox(index), onmouseleave: () => focusBox(null),
      onfocus: () => focusBox(index), onblur: () => focusBox(null),
    }, [
      el("span", { class: "detection-index", text: index + 1 }),
      el("span", { class: "detection-main" }, [
        el("span", { class: "detection-label", text: "Resíduo" }),
        el("span", { class: "detection-bar" }, [el("span", { style: { width: `${item.confidence * 100}%` } })]),
      ]),
      el("span", { class: "detection-score", text: percent.format(item.confidence) }),
      el("span", { class: "detection-coords", text: `x ${x1} a ${x2}, y ${y1} a ${y2} px` }),
    ]);
    list.append(row);
  });
}

function renderVideoResult(result) {
  const s = result.summary;
  const stats = [
    ["Quadros", s.frames === null ? "—" : decimal(0).format(s.frames)],
    ["Duração", s.duration_seconds === null ? "—" : `${decimal(1).format(s.duration_seconds)} s`],
    ["Taxa do arquivo", s.source_fps === null ? "—" : `${decimal(1).format(s.source_fps)} FPS`],
    ["Processamento", s.processing_fps === null ? "—" : `${decimal(1).format(s.processing_fps)} FPS`],
    ["Latência mediana", s.p50_ms === null ? "—" : `${decimal(1).format(s.p50_ms)} ms`],
    ["Detecções", s.detections === null ? "—" : decimal(0).format(s.detections)],
  ];
  $("#video-stats").replaceChildren(...stats.map(([label, value]) =>
    el("div", {}, [el("dt", { text: label }), el("dd", { text: value })])));
  if (!result.playable) showAlert("O vídeo foi processado, mas a pré-visualização precisa do FFmpeg instalado. Baixe o vídeo anotado abaixo.");
}

function showAlert(message) { const alert = $("#alert"); alert.textContent = message; alert.hidden = false; }
function hideAlert() { $("#alert").hidden = true; }

async function analyze(kind) {
  const file = state.files[kind];
  if (!file || state.busy) return;
  state.busy = true;
  hideAlert();
  const button = $(kind === "image" ? "#analyze-image" : "#analyze-video");
  const label = button.textContent;
  button.classList.add("is-busy");
  button.textContent = kind === "image" ? "Analisando imagem…" : "Processando vídeo…";
  renderStage();
  const body = new FormData();
  body.append("file", file, state.names[kind] || file.name);
  body.append("confidence", confidence.value);
  try {
    const response = await fetch(kind === "image" ? "/api/image" : "/api/video", { method: "POST", body });
    const data = await response.json().catch(() => ({ error: "A resposta do servidor não pôde ser lida." }));
    renderStatus(data.model);
    if (!response.ok) throw new Error(data.error || "O processamento falhou.");
    state.results[kind] = data;
  } catch (error) {
    showAlert(error.message === "Failed to fetch"
      ? "Sem conexão com o servidor local. Confira se a aplicação continua aberta no terminal."
      : error.message);
  } finally {
    state.busy = false;
    button.classList.remove("is-busy");
    button.textContent = label;
    renderStage();
    renderResult();
  }
}
$("#analyze-image").addEventListener("click", () => analyze("image"));
$("#analyze-video").addEventListener("click", () => analyze("video"));

async function loadSamples() {
  let samples = [];
  try { samples = await getJSON("/api/samples"); } catch { return; }
  if (!samples.length) return;
  const row = $("#sample-row");
  for (const sample of samples) {
    const thumb = sample.kind === "image"
      ? el("span", { class: "sample-thumb" }, [el("img", { src: sample.url, alt: "", loading: "lazy" })])
      : el("span", { class: "sample-thumb is-video" }, [el("span", { class: "play", "aria-hidden": "true" }), "Vídeo"]);
    row.append(el("button", {
      class: "sample", type: "button", "aria-pressed": "false", "data-name": sample.name, "data-kind": sample.kind,
      onclick: async () => {
        try {
          const response = await fetch(sample.url);
          const blob = await response.blob();
          setFile(sample.kind, new File([blob], sample.name, { type: blob.type }), sample.name);
        } catch {
          showAlert("Não foi possível abrir a amostra.");
        }
      },
    }, [thumb, el("span", { class: "sample-title" }, [sample.title,
      el("span", { class: "sample-kind", text: sample.kind === "image" ? "Imagem do teste" : "Montagem, não é captura real" })])]));
  }
  $("#samples").hidden = false;
}

/* ---------- Sobre o modelo ---------- */

let projectLoaded = false;

async function loadProject() {
  if (projectLoaded) return;
  projectLoaded = true;
  let data;
  try { data = await getJSON("/api/project"); } catch {
    projectLoaded = false;
    $("#doc-issues").textContent = "Não foi possível carregar os registros do projeto.";
    $("#doc-issues").hidden = false;
    return;
  }
  $("#doc-model").replaceChildren("Modelo avaliado: ", el("strong", { text: data.model.name }), ", SHA-256 ", el("code", { text: data.model.sha256 }));
  if (data.issues.length) {
    $("#doc-issues").textContent = `Alguns registros estão indisponíveis: ${data.issues.join(", ")}. Os campos correspondentes aparecem como não registrados.`;
    $("#doc-issues").hidden = false;
  }

  const ev = data.evaluation;
  $("#eval-lede").textContent = `${ev.images} imagens e ${ev.instances} objetos anotados que o modelo nunca viu no treino. Confiança mínima de ${ev.confidence} e IoU de pareamento de ${ev.iou}.`;
  const targetLeft = `${ev.target * 100}%`;
  const meters = $("#meters");
  meters.replaceChildren(el("div", { class: "meter-axis", "aria-hidden": "true" }, [
    el("span"), el("span", {}, [el("b", { style: { left: targetLeft }, text: `meta interna ${ev.target_text}` })]), el("span"),
  ]));
  for (const score of ev.scores) {
    const width = score.value === null ? 0 : Math.max(0, Math.min(1, score.value)) * 100;
    meters.append(el("div", { class: "meter", title: `${score.label}: ${score.text}. ${score.meaning}.` }, [
      el("span", { class: "meter-name" }, [score.label, el("small", { text: score.meaning })]),
      el("span", { class: "meter-track", role: "img", "aria-label": `${score.label} ${score.text}, meta ${ev.target_text}` }, [
        el("span", { class: "meter-fill", style: { width: `${width}%` } }),
        el("span", { class: "meter-target", style: { left: targetLeft } }),
      ]),
      el("span", { class: "meter-value", text: score.text }),
    ]));
  }
  $("#counts").replaceChildren(...ev.counts.map((item) =>
    el("dl", { class: "count" }, [el("dt", { text: item.label }), el("dd", { text: item.text })])));
  if (ev.below_target) {
    $("#target-note").textContent = `Precisão, recall e AP50 ficaram abaixo da meta interna de ${ev.target_text}. Os resultados não sustentam uma alegação de validação operacional. Use as detecções como apoio, sempre com conferência humana.`;
    $("#target-note").hidden = false;
  }

  const splits = data.dataset.splits;
  const total = splits.reduce((sum, item) => sum + (item.images || 0), 0) || 1;
  $("#split").replaceChildren(...splits.map((item) =>
    el("span", { style: { flex: `${item.images || 0} 0 0` }, title: `${item.label}: ${item.images_text} imagens` })));
  $("#split").setAttribute("role", "img");
  $("#split").setAttribute("aria-label", splits.map((item) => `${item.label} ${item.images_text} imagens`).join(", "));
  const swatches = ["var(--moss)", "color-mix(in srgb, var(--moss) 60%, var(--surface))", "color-mix(in srgb, var(--moss) 35%, var(--surface))"];
  $("#split-table tbody").replaceChildren(...splits.map((item, index) => el("tr", {}, [
    el("td", {}, [el("span", { class: "swatch", style: { background: swatches[index] } }), item.label,
      el("span", { class: "small", text: item.images ? `  ${percent.format(item.images / total)}` : "" })]),
    el("td", { class: "num", text: item.images_text }),
    el("td", { class: "num", text: item.instances_text }),
  ])));

  $("#training-facts").replaceChildren(...data.training.map(([label, value]) =>
    el("div", {}, [el("dt", { text: label }), el("dd", { text: value })])));
  $("#experiments tbody").replaceChildren(...data.experiments.map((item) => el("tr", {}, [
    el("td", { text: item.name }), el("td", { class: "num", text: item.epochs }),
    el("td", { class: "num", text: item.batch }), el("td", { class: "num", text: item.map50 }),
  ])));
  $("#resolution tbody").replaceChildren(...data.resolution.candidates.map((item) => el("tr", {}, [
    el("td", { text: `${item.imgsz} px` + (item.imgsz === data.resolution.selected ? " (escolhida)" : "") }),
    el("td", { class: "num", text: item.map50 }), el("td", { class: "num", text: item.map50_95 }),
  ])));
  $("#resolution-note").textContent = "A entrada maior só seria adotada se o AP50-95 melhorasse mais de 0,005 na validação.";

  const figureMeta = {
    results: ["Perdas e métricas ao longo das 60 épocas, no treino e na validação.", true],
    pr: ["Precisão × recall na validação.", false],
    f1: ["F1 por limiar de confiança na validação.", false],
    comparison: ["Referência de uma época contra o ajuste completo, na mesma validação.", false],
  };
  $("#figures").replaceChildren(...Object.entries(figureMeta)
    .filter(([key]) => data.figures.includes(key))
    .map(([key, [caption, wide]]) => el("figure", { class: "figure" + (wide ? " is-wide" : "") }, [
      el("img", { src: `/figures/${key}`, alt: caption, loading: "lazy" }),
      el("figcaption", { text: caption }),
    ])));

  $("#benchmark").replaceChildren(...data.benchmark.map(([label, value]) =>
    el("div", {}, [el("dt", { text: label }), el("dd", { text: value })])));

  renderCases(data.errors);

  $("#team").replaceChildren(...data.team.map((member) =>
    el("li", {}, [el("strong", { text: member.name }), el("span", { text: `RM ${member.rm}` })])));

  const linkNames = { report: "Relatório técnico", video: "Vídeo da demonstração", repository: "Código no GitHub", weights: "Checkpoint (best.pt)" };
  $("#links").replaceChildren(...Object.entries(linkNames).filter(([key]) => data.links[key]).map(([key, label]) =>
    el("a", { class: "button button-line", href: data.links[key], target: "_blank", rel: "noopener", text: label })));
  if (data.links.report) {
    $("#doc-report").href = data.links.report;
    $("#doc-report").hidden = false;
  }
  $("#limitations").replaceChildren(...data.limitations.map((item) => el("li", { text: item })));
  watchSections();
}

function renderCases(cases) {
  const list = $("#case-list");
  if (!cases.length) {
    $("#case").textContent = "Nenhum caso de erro registrado.";
    return;
  }
  const select = (index) => {
    for (const tab of list.children) tab.setAttribute("aria-selected", String(Number(tab.dataset.index) === index));
    const item = cases[index];
    const panel = $("#case");
    panel.setAttribute("aria-labelledby", `case-tab-${index}`);
    panel.replaceChildren(
      item.figure
        ? el("figure", { class: "case-figure" }, [
          el("img", { src: item.figure, alt: `Imagem ${item.image} com caixas de referência e predições` }),
          el("figcaption", { text: "Caixas de referência em azul, predições em verde." })])
        : el("p", { class: "small", text: "A fotografia deste caso não foi incluída na apresentação. Os resultados quantitativos permanecem registrados." }),
      el("dl", { class: "case-counts" }, [
        el("div", {}, [el("dt", { text: "Acertos" }), el("dd", { text: item.true_positives })]),
        el("div", {}, [el("dt", { text: "Falsos positivos" }), el("dd", { text: item.false_positives })]),
        el("div", {}, [el("dt", { text: "Não encontrados" }), el("dd", { text: item.false_negatives })]),
      ]),
      el("dl", { class: "case-notes" }, [
        el("div", {}, [el("dt", { text: "Observação" }), el("dd", { text: item.observation })]),
        el("div", {}, [el("dt", { text: "Hipótese" }), el("dd", { text: item.hypothesis })]),
        el("div", {}, [el("dt", { text: "Correção proposta" }), el("dd", { text: item.correction })]),
      ]),
    );
  };
  list.replaceChildren(...cases.map((item, index) => el("button", {
    class: "case-tab", type: "button", role: "tab", id: `case-tab-${index}`, "data-index": index,
    "aria-selected": "false", onclick: () => select(index),
    onkeydown: (event) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = (index + step + cases.length) % cases.length;
      select(next);
      list.children[next].focus();
    },
  }, [el("strong", { text: `Imagem ${item.image.replace(/\.jpg$/i, "")}` }),
    el("span", { text: `${item.false_negatives} não encontrados, ${item.false_positives} falsos positivos` })])));
  const initial = Math.max(0, cases.findIndex((item) => item.figure));
  select(initial);
}

function watchSections() {
  const links = [...document.querySelectorAll(".toc a")];
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) link.classList.toggle("is-current", link.getAttribute("href") === `#${entry.target.id}`);
    }
  }, { rootMargin: "-30% 0px -60% 0px" });
  for (const link of links) {
    const target = document.querySelector(link.getAttribute("href"));
    if (target) observer.observe(target);
  }
}

for (const button of document.querySelectorAll(".copy")) {
  button.addEventListener("click", async () => {
    const text = button.parentElement.querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copiado";
    } catch {
      button.textContent = "Selecione e copie";
    }
    setTimeout(() => { button.textContent = "Copiar"; }, 1800);
  });
}

/* ---------- Início ---------- */

(async function start() {
  showView();
  try {
    const model = await getJSON("/api/status");
    renderStatus(model);
    state.defaultConfidence = model.confidence;
    confidence.value = model.confidence;
  } catch {
    $("#model-status .status-text").textContent = "Servidor indisponível";
  }
  setConfidenceLabel();
  renderStage();
  loadSamples();
})();
