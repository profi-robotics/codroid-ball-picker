(function () {
  const settingsState = {current: {}};

  function notify(message, type = "info") {
    const stack = document.getElementById("notification-stack");
    if (!stack) return;
    const item = document.createElement("div");
    item.className = `notification ${type}`;
    item.textContent = message;
    stack.appendChild(item);
    window.setTimeout(() => item.remove(), type === "error" ? 6500 : 3500);
  }

  function parseValue(node) {
    if (node.type === "checkbox") return Boolean(node.checked);
    if (node.type === "number" || node.type === "range") {
      return node.value === "" ? null : Number(node.value);
    }
    if (node.dataset.setting && ["auto_target_labels", "camera_safety_labels"].includes(node.dataset.setting)) {
      return node.value.split(",").map((item) => item.trim()).filter(Boolean);
    }
    if (node.dataset.autoConfig === "target_labels") {
      return node.value.split(",").map((item) => item.trim()).filter(Boolean);
    }
    return node.value === "" ? null : node.value;
  }

  function setByPath(target, path, value) {
    const parts = path.split(".");
    let current = target;
    parts.forEach((part, index) => {
      const isLast = index === parts.length - 1;
      const nextPart = parts[index + 1];
      if (isLast) {
        current[part] = value;
        return;
      }
      if (current[part] == null) {
        current[part] = /^\d+$/.test(nextPart) ? [] : {};
      }
      current = current[part];
    });
  }

  function valueByPath(source, path) {
    return path.split(".").reduce((current, part) => {
      if (current == null) return undefined;
      return current[part];
    }, source);
  }

  function payloadFor(button) {
    const source = button.dataset.payloadSource;
    if (source === "camera-settings") return cameraPayload();
    if (source === "robot-settings") return robotPayload();
    if (source === "gripper-settings") return gripperPayload();
    if (source === "workspace") return workspacePayload();
    if (source === "drop") return dropPayload();
    if (source === "plate-test") return plateTestPayload();
    const raw = button.dataset.payload;
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch (_err) {
      return {};
    }
  }

  async function command(button) {
    const url = button.dataset.command;
    if (!url) return;
    const previous = button.textContent;
    button.disabled = true;
    try {
      const response = await fetch(url, {
        method: button.dataset.method || "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payloadFor(button))
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.ok === false) {
        throw new Error(data.message || response.statusText);
      }
      notify(data.message || "Command complete.", "ok");
      button.textContent = "Done";
      window.setTimeout(() => { button.textContent = previous; }, 900);
    } catch (err) {
      notify(err.message || "Command failed.", "error");
      button.textContent = "Error";
      console.error(err);
      window.setTimeout(() => { button.textContent = previous; }, 1400);
    } finally {
      button.disabled = false;
    }
  }

  function tabs() {
    document.querySelectorAll("[data-tab-target]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.tabTarget;
        document.querySelectorAll("[data-tab-target]").forEach((node) => node.classList.remove("active"));
        document.querySelectorAll("[data-tab]").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        const pane = document.querySelector(`[data-tab="${target}"]`);
        if (pane) pane.classList.add("active");
        localStorage.setItem("codroid.activeTab", target);
      });
    });
    document.querySelectorAll("[data-subtab-target]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.subtabTarget;
        document.querySelectorAll("[data-subtab-target]").forEach((node) => node.classList.remove("active"));
        document.querySelectorAll("[data-subtab]").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        const pane = document.querySelector(`[data-subtab="${target}"]`);
        if (pane) pane.classList.add("active");
        localStorage.setItem("codroid.activeSubtab", target);
      });
    });
    const saved = localStorage.getItem("codroid.activeTab");
    if (saved) document.querySelector(`[data-tab-target="${saved}"]`)?.click();
    const savedSubtab = localStorage.getItem("codroid.activeSubtab");
    if (savedSubtab) document.querySelector(`[data-subtab-target="${savedSubtab}"]`)?.click();
  }

  function language() {
    const select = document.getElementById("language-select");
    if (!select) return;
    select.addEventListener("change", async () => {
      await fetch("/locale", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({language: select.value})
      });
      window.location.href = `/?lang=${encodeURIComponent(select.value)}`;
    });
  }

  async function loadSettings() {
    const response = await fetch("/settings");
    const payload = await response.json();
    const settings = payload.settings || payload;
    settingsState.current = settings;
    document.querySelectorAll("[data-setting]").forEach((node) => {
      const value = valueByPath(settings, node.dataset.setting);
      if (value == null) return;
      if (node.type === "checkbox") {
        node.checked = Boolean(value);
      } else if (Array.isArray(value)) {
        node.value = value.join(",");
      } else {
        node.value = value;
      }
    });
    document.querySelectorAll("[data-detection-param]").forEach((node) => {
      const key = node.dataset.detectionParam;
      const detection = window.codroidState?.current?.detection || {};
      if (detection[key] != null) node.value = detection[key];
    });
    document.querySelectorAll("[data-auto-config]").forEach((node) => {
      const key = node.dataset.autoConfig;
      const autoPick = window.codroidState?.current?.auto_pick || {};
      const fallbackMap = {drop_ignore_radius_mm: "auto_drop_ignore_radius_mm"};
      const value = autoPick[key] ?? settings[fallbackMap[key] || key];
      if (Array.isArray(value)) node.value = value.join(",");
      else if (value != null) node.value = value;
    });
    window.hydrateI18n?.();
  }

  function collectSettings() {
    const updates = {};
    document.querySelectorAll("[data-setting]").forEach((node) => {
      setByPath(updates, node.dataset.setting, parseValue(node));
    });
    return updates;
  }

  async function saveSettings() {
    const response = await fetch("/settings", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({updates: collectSettings()})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || "Failed to save settings.");
    notify(data.message || "Settings saved.", "ok");
    await loadSettings();
  }

  function cameraPayload() {
    const settings = collectSettings();
    const resolution = settings.camera_resolution || [];
    return {
      source: settings.camera_source || settingsState.current.camera_source,
      width: Number(resolution[0] || settingsState.current.camera_resolution?.[0] || 1280),
      height: Number(resolution[1] || settingsState.current.camera_resolution?.[1] || 720),
      fps: Number(settings.camera_fps || settingsState.current.camera_fps || 30),
      host_camera_index: Number(settings.host_camera_index || 0),
      host_camera_device: settings.host_camera_device || null,
      enable_depth: Boolean(settings.camera_enable_depth),
      lock_af: Boolean(document.getElementById("camera-lock-af")?.checked)
    };
  }

  function robotPayload() {
    const settings = collectSettings();
    return {
      host: settings.robot_host || settingsState.current.robot_host || null,
      robot_port: settings.robot_port || settingsState.current.robot_port || null,
      user_port: settings.user_port || settingsState.current.user_port || null
    };
  }

  function gripperPayload() {
    const settings = collectSettings();
    return {
      ip: settings.gripper_ip || settingsState.current.gripper_ip || null,
      type: settings.gripper_type || settingsState.current.gripper_type || "auto"
    };
  }

  function workspacePayload() {
    const settings = collectSettings();
    return {
      marker_ids: settings.marker_ids || settingsState.current.marker_ids || [1, 2],
      marker_size_mm: Number(settings.marker_size_mm || settingsState.current.marker_size_mm || 40)
    };
  }

  function dropPayload() {
    return {drop_height_mm: Number(document.getElementById("drop-height")?.value || 10)};
  }

  function plateTestPayload() {
    const payload = {};
    document.querySelectorAll("[data-plate-test]").forEach((node) => {
      payload[node.dataset.plateTest] = parseValue(node);
    });
    return payload;
  }

  function collectDetectionParams() {
    const updates = {};
    document.querySelectorAll("[data-detection-param]").forEach((node) => {
      updates[node.dataset.detectionParam] = parseValue(node);
    });
    return updates;
  }

  async function saveDetectionParams() {
    const response = await fetch("/detection/params", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectDetectionParams())
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || "Failed to update detection params.");
    notify(data.message || "Detection params updated.", "ok");
  }

  function collectAutoConfig() {
    const updates = {};
    document.querySelectorAll("[data-auto-config]").forEach((node) => {
      updates[node.dataset.autoConfig] = parseValue(node);
    });
    return updates;
  }

  async function saveAutoConfig() {
    const updates = collectAutoConfig();
    const response = await fetch("/auto/config", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target_labels: updates.target_labels})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || "Failed to update auto config.");
    await fetch("/settings", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        updates: {
          auto_target_labels: updates.target_labels,
          auto_drop_ignore_radius_mm: updates.drop_ignore_radius_mm
        }
      })
    });
    notify(data.message || "Auto config updated.", "ok");
  }

  async function refreshModels({silent = false} = {}) {
    const response = await fetch("/models");
    const data = await response.json();
    const select = document.getElementById("model-select");
    if (select) {
      select.innerHTML = "";
      (data.models || []).forEach((model) => {
        const option = document.createElement("option");
        option.value = model.path || model;
        option.textContent = model.name || model.path || model;
        if (option.value === data.active) option.selected = true;
        select.appendChild(option);
      });
    }
    const labels = data.labels || [];
    const labelInput = document.querySelector('[data-auto-config="target_labels"]');
    if (labelInput && !labelInput.value && labels.length) labelInput.value = labels.join(",");
    if (!silent) notify(`Models refreshed (${(data.models || []).length}).`, "info");
  }

  async function uploadModel(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/models/upload", {method: "POST", body: form});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || "Model upload failed.");
    notify(data.message || "Model uploaded.", "ok");
    await refreshModels();
  }

  async function activateModel() {
    const path = document.getElementById("model-select")?.value || null;
    const response = await fetch("/models/activate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || "Model activation failed.");
    notify(data.message || "Model activation updated.", "ok");
  }

  async function benchmarkModel() {
    const response = await fetch("/models/benchmark", {method: "POST"});
    const data = await response.json().catch(() => ({}));
    notify(data.message || `Benchmark ${data.status || "complete"}.`, data.status === "unavailable" ? "info" : "ok");
  }

  async function scanCameras() {
    const response = await fetch("/camera/devices");
    const data = await response.json();
    const list = document.getElementById("camera-device-list");
    if (!list) return;
    list.innerHTML = "";
    (data.devices || []).forEach((device) => {
      const item = document.createElement("div");
      item.className = "list-item";
      item.textContent = `${device.label || device.name || "Camera"} ${device.index != null ? `#${device.index}` : ""}`;
      list.appendChild(item);
    });
    if (!list.children.length) list.textContent = "No cameras reported by scanner.";
  }

  function wireSliders() {
    document.querySelectorAll("input[type='range']").forEach((input) => {
      const update = (send = false) => {
        document.querySelectorAll(`[data-output-for="${input.id}"]`).forEach((node) => { node.textContent = input.value; });
        if (send && input.id === "robot-speed-slider") {
          const robot = window.codroidState?.current?.robot || {};
          if (robot.status !== "online") {
            notify("Robot is not connected.", "error");
            return;
          }
          fetch("/robot/speed", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({rate: Number(input.value) / 100})
          }).catch((err) => notify(err.message || "Failed to set robot speed.", "error"));
        }
      };
      input.addEventListener("input", () => update(true));
      update(false);
    });
    document.querySelectorAll("[data-step-input]").forEach((button) => {
      button.addEventListener("click", () => {
        const input = document.getElementById(button.dataset.stepInput);
        if (!input) return;
        input.value = String(Math.min(Number(input.max || 100), Math.max(Number(input.min || 0), Number(input.value || 0) + Number(button.dataset.step || 1))));
        input.dispatchEvent(new Event("input", {bubbles: true}));
      });
    });
  }

  function wireFooterCommands() {
    document.getElementById("robot-power-toggle")?.addEventListener("change", (event) => {
      fetch("/robot/power", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({on: event.target.checked})
      }).catch((err) => notify(err.message, "error"));
    });
    document.getElementById("footer-gripper-open")?.addEventListener("click", () => {
      fetch("/gripper/open", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({width_mm: Number(document.getElementById("footer-gripper-width")?.value || 80), speed: Number(document.getElementById("footer-gripper-speed")?.value || 80)})
      }).then(() => notify("Gripper open requested.", "ok")).catch((err) => notify(err.message, "error"));
    });
    document.getElementById("footer-gripper-close")?.addEventListener("click", () => {
      fetch("/gripper/close", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          width_mm: Number(document.getElementById("footer-gripper-width")?.value || 20),
          force: Number(document.getElementById("footer-gripper-force")?.value || 40),
          speed: Number(document.getElementById("footer-gripper-speed")?.value || 80)
        })
      }).then(() => notify("Gripper close requested.", "ok")).catch((err) => notify(err.message, "error"));
    });
    document.getElementById("apply-gripper-orientation")?.addEventListener("click", () => {
      fetch("/gripper/orientation", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({orientation: document.getElementById("gripper-orientation")?.value || "inward"})
      }).then(() => notify("Gripper orientation updated.", "ok")).catch((err) => notify(err.message, "error"));
    });
  }

  function dialogs() {
    const setup = document.getElementById("setup-dialog");
    document.getElementById("open-setup")?.addEventListener("click", () => setup?.showModal());
    document.getElementById("footer-toggle")?.addEventListener("click", () => {
      document.getElementById("app-footer")?.classList.toggle("collapsed");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    tabs();
    language();
    dialogs();
    wireSliders();
    wireFooterCommands();
    document.querySelectorAll("[data-command]").forEach((button) => {
      button.addEventListener("click", () => command(button));
    });
    document.getElementById("settings-save")?.addEventListener("click", () => saveSettings().catch((err) => notify(err.message, "error")));
    document.getElementById("save-detection-params")?.addEventListener("click", () => saveDetectionParams().catch((err) => notify(err.message, "error")));
    document.getElementById("save-auto-config")?.addEventListener("click", () => saveAutoConfig().catch((err) => notify(err.message, "error")));
    document.getElementById("camera-scan")?.addEventListener("click", () => scanCameras().catch((err) => notify(err.message, "error")));
    document.getElementById("model-upload-input")?.addEventListener("change", (event) => uploadModel(event.target.files?.[0]).catch((err) => notify(err.message, "error")));
    document.getElementById("load-model")?.addEventListener("click", () => activateModel().catch((err) => notify(err.message, "error")));
    document.getElementById("benchmark-model")?.addEventListener("click", () => benchmarkModel().catch((err) => notify(err.message, "error")));
    loadSettings().catch((err) => notify(err.message, "error"));
    refreshModels({silent: true}).catch(() => {});
  });
})();
