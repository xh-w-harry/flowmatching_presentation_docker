const API_BASE = window.API_BASE || "http://localhost:6052";
const boreholeList = document.querySelector("#boreholeList");
const resultBox = document.querySelector("#resultBox");
const downloads = document.querySelector("#downloads");
const statusEl = document.querySelector("#status");
const submitBtn = document.querySelector("#submitBtn");
const cancelBtn = document.querySelector("#cancelBtn");

let currentTaskId = null;
let pollTimer = null;

const defaults = [
  { x: 8, y: 24, azimuth: 120, vertical: 90 },
  { x: 24, y: 17, azimuth: 120, vertical: 90 },
  { x: 40, y: 35, azimuth: 300, vertical: 90 },
  { x: 56, y: 38, azimuth: 300, vertical: 90 },
];

function defaultAzimuth() {
  return Number(document.querySelector("#azimuth").value);
}

function defaultVertical() {
  return Number(document.querySelector("#vertical").value);
}

function addBoreholeRow(x = "", y = "", azimuth = defaultAzimuth(), vertical = defaultVertical()) {
  const row = document.createElement("div");
  row.className = "borehole-row";
  row.innerHTML = `
    <label><span>X</span><input class="borehole-x" type="number" min="0" step="1" value="${x}" required /></label>
    <label><span>Y</span><input class="borehole-y" type="number" min="0" step="1" value="${y}" required /></label>
    <label><span>水平角</span><input class="borehole-azimuth" type="number" min="0" max="360" step="0.1" value="${azimuth}" required /></label>
    <label><span>纵向角</span><input class="borehole-vertical" type="number" min="0" max="180" step="0.1" value="${vertical}" required /></label>
    <button type="button" title="删除">删除</button>
  `;
  row.querySelector("button").addEventListener("click", () => row.remove());
  boreholeList.appendChild(row);
}

function collectBoreholes() {
  return [...document.querySelectorAll(".borehole-row")].map((row) => ({
    x: Number(row.querySelector(".borehole-x").value),
    y: Number(row.querySelector(".borehole-y").value),
    azimuth_degrees: Number(row.querySelector(".borehole-azimuth").value),
    vertical_degrees: Number(row.querySelector(".borehole-vertical").value),
  }));
}

function setStatus(text) {
  statusEl.textContent = text;
}

function renderDownloads(data) {
  downloads.innerHTML = "";
  const zip = document.createElement("a");
  zip.href = `${API_BASE}${data.download_zip_url}`;
  zip.textContent = "下载 outputs.zip";
  zip.target = "_blank";
  downloads.appendChild(zip);

  for (const file of data.files || []) {
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "secondary";
    preview.textContent = `预览 ${file.name}`;
    preview.addEventListener("click", () => previewFile(data.run_id, file.name));
    downloads.appendChild(preview);

    const link = document.createElement("a");
    link.href = `${API_BASE}${file.url}`;
    link.textContent = `下载 ${file.name}`;
    link.target = "_blank";
    downloads.appendChild(link);
  }
}

async function previewFile(runId, filename) {
  resultBox.textContent = `加载预览：${filename}`;
  try {
    const response = await fetch(`${API_BASE}/api/runs/${runId}/preview/${encodeURIComponent(filename)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "预览失败");
    }
    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultBox.textContent = String(error);
  }
}

function startPolling(taskId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollTask(taskId), 1500);
  pollTask(taskId);
}

async function pollTask(taskId) {
  try {
    const response = await fetch(`${API_BASE}/api/tasks/${taskId}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "任务状态获取失败");
    }

    resultBox.textContent = JSON.stringify(data.metadata || data, null, 2);
    const progress = data.progress || {};
    setStatus(`${data.status} ${progress.completed_generations || 0}/${progress.num_generations || 0}`);

    if (["completed", "cancelled", "failed"].includes(data.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      submitBtn.disabled = false;
      cancelBtn.disabled = true;
      currentTaskId = null;
      if (data.metadata) {
        renderDownloads(data.metadata);
      }
    }
  } catch (error) {
    resultBox.textContent = String(error);
    clearInterval(pollTimer);
    pollTimer = null;
    submitBtn.disabled = false;
    cancelBtn.disabled = true;
  }
}

document.querySelector("#addBorehole").addEventListener("click", () => addBoreholeRow());

document.querySelector("#healthBtn").addEventListener("click", async () => {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultBox.textContent = String(error);
  }
});

cancelBtn.addEventListener("click", async () => {
  if (!currentTaskId) return;
  cancelBtn.disabled = true;
  setStatus("正在请求终止");
  try {
    const response = await fetch(`${API_BASE}/api/tasks/${currentTaskId}/cancel`, { method: "POST" });
    const data = await response.json();
    resultBox.textContent = JSON.stringify(data, null, 2);
    setStatus(data.status);
  } catch (error) {
    resultBox.textContent = String(error);
  }
});

document.querySelector("#inferForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData();
  const boreholeFile = form.elements.borehole_file.files[0];
  const geoFile = form.elements.geo_file.files[0];
  const gravityFile = form.elements.gravity_file.files[0];
  const magneticsFile = form.elements.magnetics_file.files[0];

  if (!boreholeFile) {
    setStatus("请上传 Borehole .npy");
    return;
  }
  if (!geoFile) {
    setStatus("请上传 GeoData .npy");
    return;
  }

  formData.append("borehole_file", boreholeFile);
  formData.append("geo_file", geoFile);
  if (gravityFile) formData.append("gravity_file", gravityFile);
  if (magneticsFile) formData.append("magnetics_file", magneticsFile);
  formData.append("boreholes", JSON.stringify(collectBoreholes()));
  formData.append("azimuth_degrees", document.querySelector("#azimuth").value);
  formData.append("vertical_degrees", document.querySelector("#vertical").value);
  formData.append("num_classes", document.querySelector("#numClasses").value);
  formData.append("num_generations", document.querySelector("#numGenerations").value);
  formData.append("ode_steps", document.querySelector("#odeSteps").value);
  formData.append("bounds", document.querySelector("#bounds").value);

  submitBtn.disabled = true;
  cancelBtn.disabled = true;
  downloads.innerHTML = "";
  resultBox.textContent = "任务提交中...";
  setStatus("提交任务");

  try {
    const response = await fetch(`${API_BASE}/api/infer`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "推理失败");
    }
    currentTaskId = data.task_id;
    cancelBtn.disabled = false;
    resultBox.textContent = JSON.stringify(data, null, 2);
    setStatus(`任务已提交：${currentTaskId}`);
    startPolling(currentTaskId);
  } catch (error) {
    resultBox.textContent = String(error);
    setStatus("失败");
    submitBtn.disabled = false;
    cancelBtn.disabled = true;
  }
});

for (const item of defaults) {
  addBoreholeRow(item.x, item.y, item.azimuth, item.vertical);
}
