(function () {
  "use strict";

  const DISCONNECT_GRACE_MS = 3000;

  let source = null;
  let lastSeq = 0;
  let lastSnapshot = null;
  let disconnectTimer = null;
  let pendingProps = {};
  let flushTimer = null;
  let started = false;

  function scheduleFlush() {
    if (flushTimer) return;
    flushTimer = window.setTimeout(function () {
      flushTimer = null;
      flushPendingProps();
    }, 100);
  }

  function setProps(id, props) {
    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function"
    ) {
      try {
        window.dash_clientside.set_props(id, props);
        return true;
      } catch (error) {
        pendingProps[id] = Object.assign({}, pendingProps[id] || {}, props);
        scheduleFlush();
      }
    }
    return false;
  }

  function setMountedProps(id, props) {
    if (!document.getElementById(id)) return false;
    return setProps(id, props);
  }

  function flushPendingProps() {
    const ids = Object.keys(pendingProps);
    if (!ids.length) return;
    const retry = {};
    ids.forEach(function (id) {
      try {
        window.dash_clientside.set_props(id, pendingProps[id]);
      } catch (error) {
        retry[id] = pendingProps[id];
      }
    });
    pendingProps = retry;
    if (Object.keys(pendingProps).length) {
      scheduleFlush();
    }
  }

  function setIntervalPressure(_connected) {
    // Keep Dash intervals authoritative for reliability. Some Dash component
    // paths are not registered while drawers/tabs are unmounted, and set_props
    // to those paths can disrupt unrelated controls.
  }

  function markSseState(status, connected, extra) {
    // Status is kept in JS memory; Dash fallback intervals remain enabled.
    void status;
    void extra;
    setIntervalPressure(connected);
  }

  function scheduleDisconnected() {
    if (disconnectTimer) {
      clearTimeout(disconnectTimer);
    }
    disconnectTimer = setTimeout(function () {
      markSseState("disconnected", false);
    }, DISCONNECT_GRACE_MS);
  }

  function markConnected() {
    if (disconnectTimer) {
      clearTimeout(disconnectTimer);
      disconnectTimer = null;
    }
    markSseState("connected", true);
  }

  function applyCamera(camera) {
    if (!camera) return;
    setMountedProps("camera-status-text", { children: camera.message || "" });
    setMountedProps("header-camera-badge", {
      children: camera.badge || "Camera: --",
      color: camera.badge_color || "gray",
    });
    if (typeof camera.checked === "boolean") {
      setMountedProps("camera-enable", { checked: camera.checked });
    }
  }

  function applyRobot(robot) {
    if (!robot) return;
    setMountedProps("robot-status-text", { children: robot.message || "" });
    setMountedProps("header-robot-badge", {
      children: robot.badge || "Robot: --",
      color: robot.badge_color || "gray",
    });
    if (robot.position) {
      setMountedProps("footer-robot-position", { children: robot.position });
      setMountedProps("current-robot-position", { children: robot.position });
    }
  }

  function applyGripper(gripper) {
    if (!gripper) return;
    setMountedProps("gripper-status-text", {
      children: gripper.message || "",
    });
    setMountedProps("header-gripper-badge", {
      children: gripper.badge || "Gripper: --",
      color: gripper.badge_color || "gray",
    });
  }

  function applyOperation(operation) {
    if (!operation) return;
  }

  function applySafety(safety) {
    if (!safety) return;
  }

  function applySnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== "object") return;
    const merged = Object.assign({}, lastSnapshot || {}, snapshot);
    lastSeq = merged.seq || lastSeq;
    lastSnapshot = merged;
    window.dispatchEvent(
      new CustomEvent("codroid-ui-snapshot", { detail: merged })
    );
    applyCamera(merged.camera);
    applyRobot(merged.robot);
    applyGripper(merged.gripper);
    applyOperation(merged.operation);
    applySafety(merged.safety);
    markConnected();
  }

  function connect() {
    if (source) {
      source.close();
    }
    markSseState("connecting", false);
    source = new EventSource("/ui_events");
    source.addEventListener("open", function () {
      markConnected();
    });
    source.addEventListener("snapshot", function (event) {
      try {
        applySnapshot(JSON.parse(event.data));
      } catch (error) {
        console.warn("Failed to apply UI snapshot", error);
      }
    });
    source.addEventListener("heartbeat", function (event) {
      try {
        const payload = JSON.parse(event.data);
        if (payload && payload.seq) {
          lastSeq = payload.seq;
        }
      } catch (error) {
        // Heartbeats are best-effort health hints.
      }
      markConnected();
    });
    source.addEventListener("error", function () {
      scheduleDisconnected();
    });
  }

  function startWhenDashReady() {
    if (started) return;
    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function"
    ) {
      started = true;
      window.setTimeout(connect, 500);
      return;
    }
    window.setTimeout(startWhenDashReady, 100);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startWhenDashReady);
  } else {
    startWhenDashReady();
  }
})();
