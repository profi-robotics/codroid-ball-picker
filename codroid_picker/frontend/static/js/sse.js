(function () {
  const state = {
    current: window.__INITIAL_STATE__ || {},
    connected: false
  };

  function setText(name, value) {
    document.querySelectorAll(`[data-text="${name}"]`).forEach((node) => {
      node.textContent = value == null ? "" : String(value);
    });
  }

  function setJson(name, value) {
    document.querySelectorAll(`[data-json="${name}"]`).forEach((node) => {
      node.textContent = JSON.stringify(value || {}, null, 2);
    });
  }

  function badge(name, text, cls) {
    document.querySelectorAll(`[data-badge="${name}"]`).forEach((node) => {
      node.textContent = text;
      node.classList.remove("ok", "warn", "danger");
      if (cls) node.classList.add(cls);
    });
  }

  function kv(target, rows) {
    document.querySelectorAll(`[data-monitor="${target}"]`).forEach((node) => {
      node.innerHTML = "";
      rows.forEach(([key, value, cls]) => {
        const row = document.createElement("div");
        row.className = `kv ${cls || ""}`;
        row.innerHTML = `<span>${key}</span><strong>${value == null ? "--" : value}</strong>`;
        node.appendChild(row);
      });
    });
  }

  function renderList(name, rows) {
    document.querySelectorAll(`[data-list="${name}"]`).forEach((node) => {
      node.innerHTML = "";
      rows.forEach((row) => node.appendChild(row));
    });
  }

  function pill(label, ok) {
    const node = document.createElement("span");
    node.className = `status-chip ${ok ? "ok" : "warn"}`;
    node.textContent = `${label}: ${ok ? "ok" : "missing"}`;
    return node;
  }

  function reduce(payload) {
    if (!payload) return;
    if (payload.type === "state.snapshot") {
      state.current = payload;
    } else {
      (payload.changed || []).forEach((key) => {
        state.current[key] = payload[key];
      });
      state.current.seq = payload.seq;
      state.current.ts = payload.ts;
    }
    render();
  }

  function prerequisites(camera, robot, gripper, detection, calibration, autoPick) {
    const artifacts = calibration.artifacts || {};
    return {
      camera: camera.status === "live",
      camera_calibrated: Boolean(artifacts.camera),
      workspace: calibration.workspace_status === "ready" || Boolean(artifacts.workspace),
      robot_calibrated: calibration.robot_status === "complete" || calibration.robot_verification_ready || Boolean(artifacts.robot),
      robot: robot.status === "online",
      model: Boolean(detection.active_model),
      gripper: gripper.status === "online",
      drop_target: Boolean(autoPick.drop_target)
    };
  }

  function setDisabled(prereq, operation, safety) {
    const autoReady = Object.values(prereq).every(Boolean) && !safety.blocked && operation.status !== "running";
    document.querySelectorAll('[data-requires="auto-ready"]').forEach((node) => { node.disabled = !autoReady; });
    document.querySelectorAll('[data-requires="robot-verification"]').forEach((node) => {
      node.disabled = !(state.current.calibration || {}).robot_verification_ready;
    });
  }

  function renderDetections(detection) {
    const rows = (detection.detections || []).map((item) => {
      const row = document.createElement("div");
      row.className = "list-item detection-item";
      const xy = item.coord5_mm || item.workspace_mm || item.center || [];
      row.innerHTML = `<strong>${item.label || "object"}</strong><span>${Number(item.confidence || 0).toFixed(2)}</span><small>${Array.isArray(xy) ? xy.map((v) => Number(v).toFixed(1)).join(", ") : ""}</small>`;
      return row;
    });
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "list-item muted";
      empty.textContent = "No detections.";
      rows.push(empty);
    }
    renderList("detections", rows);
  }

  function renderPrereqs(prereq) {
    const labels = {
      camera: "Camera",
      camera_calibrated: "Camera calib",
      workspace: "Workspace",
      robot_calibrated: "Robot calib",
      robot: "Robot",
      model: "Model",
      gripper: "Gripper",
      drop_target: "Drop"
    };
    const rows = Object.entries(prereq).map(([key, ok]) => pill(labels[key] || key, ok));
    renderList("prerequisites", rows);
    renderList("footer-prerequisites", rows.map((node) => node.cloneNode(true)));
  }

  function renderTrend(name, values, suffix) {
    document.querySelectorAll(`[data-trend="${name}"]`).forEach((node) => {
      node.innerHTML = "";
      const trimmed = values.slice(-24);
      const max = Math.max(...trimmed, 1);
      trimmed.forEach((value) => {
        const bar = document.createElement("i");
        bar.style.height = `${Math.max(4, (Number(value || 0) / max) * 100)}%`;
        bar.title = `${Number(value || 0).toFixed(1)}${suffix}`;
        node.appendChild(bar);
      });
      if (!trimmed.length) node.textContent = "No trend data yet.";
    });
  }

  function renderDialogs(robot) {
    const active = String(robot.active_error || "").toLowerCase();
    document.querySelectorAll("[data-error-dialog]").forEach((dialog) => {
      const matches = active && active.includes(dialog.dataset.errorDialog);
      if (matches && !dialog.open) dialog.showModal();
      if (!matches && dialog.open) dialog.close();
    });
  }

  function render() {
    const camera = state.current.camera || {};
    const robot = state.current.robot || {};
    const gripper = state.current.gripper || {};
    const safety = state.current.safety || {};
    const operation = state.current.operation || {};
    const video = state.current.video || {};
    const detection = state.current.detection || {};
    const autoPick = state.current.auto_pick || {};
    const calibration = state.current.calibration || {};
    const monitoring = state.current.monitoring || {};
    const prereq = prerequisites(camera, robot, gripper, detection, calibration, autoPick);

    badge("camera", `Camera: ${camera.status || "off"}`, camera.status === "live" ? "ok" : "warn");
    badge("robot", `Robot: ${robot.status || "offline"}`, robot.status === "online" ? "ok" : "warn");
    badge("gripper", `Gripper: ${gripper.status || "offline"}`, gripper.status === "online" ? "ok" : "warn");
    badge("safety", safety.blocked ? "Safety: blocked" : "Safety: clear", safety.blocked ? "danger" : "ok");
    badge("sse", state.connected ? "SSE: live" : "SSE: reconnecting", state.connected ? "ok" : "warn");

    setText("camera-message", camera.last_error || `${camera.status || "off"} ${camera.capture_fps ? camera.capture_fps.toFixed(1) + " FPS" : ""}`);
    setText("camera-status-text", camera.last_error || `${camera.device_id || camera.source || "camera"} ${camera.status || "off"}`);
    setText("video-stats", `MJPEG clients ${video.clients || 0}, render ${Number(video.stream_fps || 0).toFixed(1)} FPS`);
    setText("operation-message", operation.message || "Program idle.");
    setText("operation-message-inline", operation.message || "Program idle.");
    setText("auto-message", autoPick.message || "Auto idle.");
    setText("calibration-message", calibration.message || "Calibration idle.");
    setText("camera-calibration-state", `${calibration.camera_status || "idle"} ${Number(calibration.camera_progress || 0).toFixed(0)}%`);
    setText("workspace-calibration-state", calibration.workspace_status || "missing");
    setText("robot-calibration-state", calibration.message || calibration.robot_status || "idle");
    setText("workspace-metrics", `Workspace: ${calibration.workspace_status || "missing"}`);
    setText("model-status", detection.active_model ? `Active model: ${detection.active_model}` : "No active model.");
    setText("detection-status", detection.last_error || `${detection.worker_status || "stopped"} ${detection.latency_ms ? Number(detection.latency_ms).toFixed(1) + " ms" : ""}`);
    setText("detection-frame-skip", detection.frame_skip || 3);
    setText("footer-gripper-status", `${gripper.display_name || "Gripper"} ${gripper.status || "offline"}${gripper.width_mm != null ? `, ${Number(gripper.width_mm).toFixed(1)} mm` : ""}`);
    setText("gripper-status-text", gripper.message || gripper.status || "offline");
    setText("robot-status-text", robot.message || robot.status || "offline");
    setText("settings-status", (state.current.settings || {}).reconnect_required?.length ? `Reconnect required: ${(state.current.settings || {}).reconnect_required.join(", ")}` : "Settings ready.");
    setText("drop-position-metrics", autoPick.drop_target ? `Drop target: X=${Number(autoPick.drop_target.x || autoPick.drop_target.robot_x || 0).toFixed(1)} Y=${Number(autoPick.drop_target.y || autoPick.drop_target.robot_y || 0).toFixed(1)}` : "No drop target taught.");
    setText("overlay-status", `Overlay: ${(video.overlay_mode || "auto").replace("_", " ")}`);
    setText("plate-test-status", monitoring.plate_test?.message || monitoring.plate_test?.status || "Plate test idle.");

    if (robot.posture) {
      setText("robot-pose", `X=${Number(robot.posture.x || 0).toFixed(1)} Y=${Number(robot.posture.y || 0).toFixed(1)} Z=${Number(robot.posture.z || 0).toFixed(1)} C=${Number(robot.posture.c || 0).toFixed(1)}`);
    }

    document.querySelectorAll("[data-di]").forEach((node) => {
      const value = robot.di_state?.[node.dataset.di] || robot.di_state?.[`di${node.dataset.di}`] || 0;
      node.classList.toggle("on", Boolean(value));
    });
    document.querySelectorAll("[data-progress='camera-calibration']").forEach((node) => {
      node.style.width = `${Math.max(0, Math.min(100, Number(calibration.camera_progress || 0)))}%`;
    });

    kv("system", [
      ["Camera", `${camera.status || "off"} ${Number(camera.capture_fps || 0).toFixed(1)} FPS`],
      ["Video", `${Number(video.stream_fps || 0).toFixed(1)} FPS, ${video.clients || 0} clients`],
      ["Robot", robot.status || "offline"],
      ["Gripper", gripper.status || "offline"]
    ]);
    if (monitoring.gpu?.available) {
      kv("system", [
        ["Camera", `${camera.status || "off"} ${Number(camera.capture_fps || 0).toFixed(1)} FPS`],
        ["Video", `${Number(video.stream_fps || 0).toFixed(1)} FPS, ${video.clients || 0} clients`],
        ["CPU", `${Number(monitoring.system?.cpu_percent || 0).toFixed(0)}%`],
        ["Memory", `${Number(monitoring.system?.memory_percent || 0).toFixed(0)}%`],
        ["GPU", `${monitoring.gpu.devices?.[0]?.name || "available"}`]
      ]);
    }
    kv("detection", [
      ["Worker", detection.worker_status || "stopped"],
      ["ML extras", detection.ml_available === false ? "missing" : (detection.ml_available === true ? "ready" : "unknown"), detection.ml_available === false ? "danger" : ""],
      ["Model", detection.active_model || "none"],
      ["Latency", detection.latency_ms ? `${Number(detection.latency_ms).toFixed(1)} ms` : "--"],
      ["Benchmark", detection.benchmark?.rating || detection.benchmark?.status || "not run"],
      ["Objects", (detection.detections || []).length]
    ]);
    kv("pick", [
      ["Attempts", monitoring.pick_attempts || 0],
      ["Successes", monitoring.pick_successes || 0],
      ["Failures", monitoring.pick_failures || 0],
      ["Last cycle", monitoring.last_cycle_s ? `${Number(monitoring.last_cycle_s).toFixed(2)} s` : "--"]
    ]);
    kv("auto-plan", [
      ["Available", autoPick.plan_snapshot?.available ? "yes" : "no"],
      ["Stage", autoPick.plan_snapshot?.stage || "--"],
      ["Mode", autoPick.plan_snapshot?.mode || "--"],
      ["Candidates", autoPick.plan_snapshot?.candidate_count || 0]
    ]);
    kv("plate", [
      ["Markers", calibration.plate?.count || 0],
      ["Stable", calibration.plate?.stable ? "yes" : "no"],
      ["Status", calibration.plate?.ok ? "visible" : "missing"],
      ["Last run", monitoring.plate_test?.status || "--"],
      ["Skipped", monitoring.plate_test?.skipped ?? "--"]
    ]);
    kv("robot-health", [
      ["Poll age", robot.poll_age_s != null ? `${Number(robot.poll_age_s).toFixed(2)} s` : "--"],
      ["Stale", robot.stale ? "yes" : "no", robot.stale ? "warn" : ""],
      ["Error", robot.active_error || "none", robot.active_error ? "danger" : ""],
      ["Safety owner", safety.owner || "--"]
    ]);
    kv("robot-calibration", [
      ["Status", calibration.robot_status || "idle"],
      ["Points", (calibration.robot_points || []).length],
      ["Verification", calibration.robot_verification_ready ? "ready" : "not ready"],
      ["Max XY error", calibration.accuracy?.max_xy_mm != null ? `${Number(calibration.accuracy.max_xy_mm).toFixed(2)} mm` : "--"]
    ]);

    renderDetections(detection);
    renderPrereqs(prereq);
    renderTrend("detection", (monitoring.detection_history || []).map((item) => item.count || 0), "");
    renderTrend("cycle", monitoring.cycle_times || [], "s");
    renderDialogs(robot);
    setDisabled(prereq, operation, safety);

    setJson("auto-plan", autoPick.plan_snapshot || {});
    setJson("robot-calibration-points", calibration.robot_points || []);
    setJson("robot-controller-frame", calibration.robot_controller_frame || {});
    setJson("detections", detection.detections || []);
    setJson("state-summary", {
      seq: state.current.seq,
      camera,
      robot,
      gripper,
      operation,
      video,
      calibration,
      auto_pick: autoPick,
      monitoring
    });
  }

  function connect() {
    const source = new EventSource("/events/status");
    source.onopen = function () {
      state.connected = true;
      render();
    };
    source.onerror = function () {
      state.connected = false;
      render();
    };
    source.addEventListener("state.snapshot", (event) => reduce(JSON.parse(event.data)));
    source.addEventListener("state.delta", (event) => reduce(JSON.parse(event.data)));
  }

  window.codroidState = state;
  window.codroidRenderState = render;
  document.addEventListener("DOMContentLoaded", () => {
    render();
    connect();
  });
})();
