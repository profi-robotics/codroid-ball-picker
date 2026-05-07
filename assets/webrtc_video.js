(function () {
  "use strict";

  const RECONNECT_AFTER_MS = 1000;
  const FRAME_STALE_AFTER_MS = 3000;

  let activeWrapper = null;
  let activePeer = null;
  let frameTimer = null;
  let scanTimer = null;
  let renderedFrames = 0;
  let fpsWindowStart = 0;
  let lastFrameAt = 0;
  let lastSnapshot = null;

  function setProps(id, props) {
    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function" &&
      document.getElementById(id)
    ) {
      try {
        window.dash_clientside.set_props(id, props);
      } catch (error) {
        // Direct DOM fallback keeps status usable before Dash has mounted props.
        const element = document.getElementById(id);
        if (element && props.children) element.textContent = props.children;
      }
    }
  }

  function setStatus(text) {
    setProps("camera-fps", { children: text });
  }

  function setMode(wrapper, mode) {
    if (!wrapper) return;
    wrapper.dataset.videoState = mode;
    const video = wrapper.querySelector("#camera-feed");
    if (video) {
      video.playsInline = true;
      video.setAttribute("playsinline", "");
      video.style.display = "block";
    }
  }

  function getPeerConnectionConstructor() {
    return (
      window.RTCPeerConnection ||
      window.webkitRTCPeerConnection ||
      window.mozRTCPeerConnection ||
      globalThis.RTCPeerConnection ||
      globalThis.webkitRTCPeerConnection ||
      globalThis.mozRTCPeerConnection ||
      null
    );
  }

  function describeWebrtcSupport() {
    return {
      hasRTCPeerConnection: Boolean(window.RTCPeerConnection),
      hasWebkitRTCPeerConnection: Boolean(window.webkitRTCPeerConnection),
      hasMozRTCPeerConnection: Boolean(window.mozRTCPeerConnection),
      hasNavigatorMediaDevices: Boolean(
        window.navigator && window.navigator.mediaDevices
      ),
      isSecureContext: Boolean(window.isSecureContext),
      protocol: window.location ? window.location.protocol : "",
      hostname: window.location ? window.location.hostname : "",
      href: window.location ? window.location.href : "",
      userAgent: window.navigator ? window.navigator.userAgent : "",
      assetVersion: "webrtc-diagnostics-2026-05-07-01",
    };
  }

  function formatSupportDiagnostic(details) {
    return (
      "rtc=" + Number(Boolean(details.hasRTCPeerConnection)) +
      " webkit=" + Number(Boolean(details.hasWebkitRTCPeerConnection)) +
      " moz=" + Number(Boolean(details.hasMozRTCPeerConnection)) +
      " secure=" + Number(Boolean(details.isSecureContext)) +
      " " + (details.protocol || "") + "//" + (details.hostname || "")
    );
  }

  function reportClientDiagnostics(event, details) {
    const payload = Object.assign(
      {
        event,
        videoElementPresent: Boolean(document.querySelector("video#camera-feed")),
        wrapperPresent: Boolean(document.getElementById("camera-video-wrapper")),
      },
      details || describeWebrtcSupport()
    );
    window.__codroidWebrtcDiagnostics = payload;
    fetch("/webrtc/client_diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(function () {
      // Diagnostics are best-effort and must never disrupt the retry path.
    });
  }

  function createVideoElement() {
    const video = document.createElement("video");
    video.id = "camera-feed";
    video.className = "camera-feed camera-feed-video";
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    return video;
  }

  function createOverlayCanvas() {
    const canvas = document.createElement("canvas");
    canvas.id = "camera-overlay-canvas";
    canvas.className = "camera-overlay-canvas";
    return canvas;
  }

  function repairWrapperContents(wrapper) {
    wrapper.replaceChildren(createVideoElement(), createOverlayCanvas());
    wrapper.dataset.repairedByAsset = "true";
    return wrapper;
  }

  function selectPrimaryWrapper() {
    const wrappers = Array.from(document.querySelectorAll("#camera-video-wrapper"));
    if (wrappers.length === 0) return null;
    const preferred = wrappers.find(function (wrapper) {
      const feed = wrapper.querySelector("#camera-feed");
      return feed && feed.tagName === "VIDEO";
    }) || wrappers.find(function (wrapper) {
      return wrapper.dataset.recoveredByAsset !== "true";
    }) || wrappers[0];

    wrappers.forEach(function (wrapper) {
      if (wrapper !== preferred) wrapper.remove();
    });
    return preferred;
  }

  function ensureVideoElements(wrapper) {
    if (!wrapper) return null;
    const existingVideo = wrapper.querySelector("#camera-feed");
    if (existingVideo && existingVideo.tagName === "VIDEO") {
      existingVideo.playsInline = true;
      existingVideo.setAttribute("playsinline", "");
      existingVideo.classList.add("camera-feed", "camera-feed-video");
      if (!wrapper.querySelector("#camera-overlay-canvas")) {
        const canvas = document.createElement("canvas");
        canvas.id = "camera-overlay-canvas";
        canvas.className = "camera-overlay-canvas";
        wrapper.appendChild(canvas);
      }
      return existingVideo;
    }

    repairWrapperContents(wrapper);
    return wrapper.querySelector("#camera-feed");
  }

  async function recoverMissingWrapper() {
    const existingWrapper = selectPrimaryWrapper();
    if (existingWrapper) {
      ensureVideoElements(existingWrapper);
      return existingWrapper;
    }
    return null;
  }

  function stopPeer() {
    if (activePeer) {
      try {
        activePeer.getSenders().forEach(function (sender) {
          if (sender.track) sender.track.stop();
        });
        activePeer.close();
      } catch (error) {
        console.warn("Failed to close WebRTC peer", error);
      }
    }
    activePeer = null;
    if (frameTimer) {
      window.clearInterval(frameTimer);
      frameTimer = null;
    }
  }

  function scheduleReconnect(reason) {
    stopPeer();
    setMode(activeWrapper, "disconnected");
    setStatus(
      "Camera: WebRTC disconnected" +
        (reason ? " (" + reason + ")" : "") +
        " · retrying"
    );
    window.setTimeout(scan, RECONNECT_AFTER_MS);
  }

  function waitForIceGathering(peer) {
    if (peer.iceGatheringState === "complete") {
      return Promise.resolve();
    }
    return new Promise(function (resolve) {
      const timeout = window.setTimeout(resolve, 2000);
      peer.addEventListener("icegatheringstatechange", function onChange() {
        if (peer.iceGatheringState === "complete") {
          window.clearTimeout(timeout);
          peer.removeEventListener("icegatheringstatechange", onChange);
          resolve();
        }
      });
    });
  }

  async function negotiate(peer) {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGathering(peer);
    const response = await fetch("/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(peer.localDescription),
    });
    if (!response.ok) {
      const error = await response.json().catch(function () {
        return {};
      });
      throw new Error(error.error || "offer failed");
    }
    const answer = await response.json();
    await peer.setRemoteDescription(answer);
  }

  function startFrameStats(video) {
    renderedFrames = 0;
    fpsWindowStart = performance.now();
    lastFrameAt = 0;

    function onFrame() {
      renderedFrames += 1;
      lastFrameAt = performance.now();
      if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback(onFrame);
      }
    }

    if (typeof video.requestVideoFrameCallback === "function") {
      video.requestVideoFrameCallback(onFrame);
    } else {
      video.addEventListener("timeupdate", function () {
        renderedFrames += 1;
        lastFrameAt = performance.now();
      });
    }

    frameTimer = window.setInterval(function () {
      const now = performance.now();
      const fps = renderedFrames * 1000 / Math.max(now - fpsWindowStart, 1);
      const stale = lastFrameAt > 0 && now - lastFrameAt > FRAME_STALE_AFTER_MS;
      const label = stale ? "WebRTC stale" : "WebRTC";
      setStatus("Camera: " + label + " · Render: " + fps.toFixed(1) + " FPS");
      renderedFrames = 0;
      fpsWindowStart = now;
      if (stale) scheduleReconnect("video_frame_stale");
    }, 1000);
  }

  function resizeOverlay(wrapper) {
    const video = wrapper.querySelector("#camera-feed");
    const canvas = wrapper.querySelector("#camera-overlay-canvas");
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 1);
    const height = Math.max(Math.floor(rect.height), 1);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
  }

  function drawOverlay(wrapper) {
    const canvas = wrapper.querySelector("#camera-overlay-canvas");
    const video = wrapper.querySelector("#camera-feed");
    if (!canvas || !video || !lastSnapshot) return;
    resizeOverlay(wrapper);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const detections = lastSnapshot.detections || {};
    const items = Array.isArray(detections.items) ? detections.items : [];
    const naturalWidth = video.videoWidth || canvas.width;
    const naturalHeight = video.videoHeight || canvas.height;
    const scaleX = canvas.width / Math.max(naturalWidth, 1);
    const scaleY = canvas.height / Math.max(naturalHeight, 1);

    ctx.lineWidth = 2;
    ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    items.slice(0, 12).forEach(function (item) {
      const bbox = item && item.bbox;
      if (!Array.isArray(bbox) || bbox.length < 4) return;
      const x = bbox[0] * scaleX;
      const y = bbox[1] * scaleY;
      const w = (bbox[2] - bbox[0]) * scaleX;
      const h = (bbox[3] - bbox[1]) * scaleY;
      const label = String(item.label || "object");
      const confidence = Number(item.confidence || 0);
      ctx.strokeStyle = "#18a058";
      ctx.fillStyle = "rgba(24, 160, 88, 0.88)";
      ctx.strokeRect(x, y, w, h);
      const text = label + (confidence ? " " + Math.round(confidence * 100) + "%" : "");
      const textWidth = ctx.measureText(text).width + 8;
      ctx.fillRect(x, Math.max(y - 20, 0), textWidth, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x + 4, Math.max(y - 7, 13));
    });
  }

  async function startWebrtc(wrapper) {
    if (activeWrapper === wrapper && activePeer) return;
    const PeerConnection = getPeerConnectionConstructor();
    if (!PeerConnection) {
      const details = describeWebrtcSupport();
      console.warn("WebRTC unavailable in this browser context", details);
      reportClientDiagnostics("webrtc_unsupported", details);
      activeWrapper = wrapper;
      scheduleReconnect("webrtc_unsupported: " + formatSupportDiagnostic(details));
      return;
    }

    stopPeer();
    activeWrapper = wrapper;

    const video = ensureVideoElements(wrapper);
    if (!video) {
      scheduleReconnect("video_element_missing");
      return;
    }

    setMode(wrapper, "connecting");
    setStatus("Camera: WebRTC connecting");
    reportClientDiagnostics("webrtc_connecting", describeWebrtcSupport());

    const peer = new PeerConnection({ iceServers: [] });
    activePeer = peer;
    let connected = false;
    const reconnectTimer = window.setTimeout(function () {
      if (!connected) scheduleReconnect("webrtc_timeout");
    }, FRAME_STALE_AFTER_MS);

    peer.addTransceiver("video", { direction: "recvonly" });
    peer.addEventListener("track", function (event) {
      if (event.track.kind !== "video") return;
      video.srcObject = event.streams[0] || new MediaStream([event.track]);
      video.play().catch(function () {
        // Autoplay is muted and should be allowed; fallback timer handles failures.
      });
    });
    peer.addEventListener("connectionstatechange", function () {
      const state = peer.connectionState;
      wrapper.dataset.webrtcConnectionState = state;
      if (state === "connected") {
        connected = true;
        window.clearTimeout(reconnectTimer);
        setMode(wrapper, "connected");
        startFrameStats(video);
      } else if (state === "failed" || state === "closed" || state === "disconnected") {
        scheduleReconnect("webrtc_" + state);
      }
    });

    try {
      await negotiate(peer);
    } catch (error) {
      console.warn("WebRTC negotiation failed", error);
      window.clearTimeout(reconnectTimer);
      scheduleReconnect(error.message || "negotiation_failed");
    }
  }

  async function scan() {
    let wrapper = selectPrimaryWrapper();
    if (!wrapper) {
      wrapper = await recoverMissingWrapper();
    }
    if (!wrapper) {
      stopPeer();
      activeWrapper = null;
      return;
    }
    if (wrapper.dataset.webrtcPreferred === "true") {
      startWebrtc(wrapper);
    }
  }

  function startScanning() {
    scan();
    if (scanTimer) return;
    scanTimer = window.setInterval(scan, 500);
  }

  function attachObserver() {
    if (!document.body) {
      window.setTimeout(attachObserver, 100);
      return;
    }
    observer.observe(document.body, { childList: true, subtree: true });
    startScanning();
  }

  window.addEventListener("codroid-ui-snapshot", function (event) {
    lastSnapshot = event.detail || null;
    if (activeWrapper && activeWrapper.dataset.videoState === "connected") {
      drawOverlay(activeWrapper);
    }
  });
  window.addEventListener("resize", function () {
    if (activeWrapper) drawOverlay(activeWrapper);
  });
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") scan();
  });

  const observer = new MutationObserver(scan);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachObserver);
  } else {
    attachObserver();
  }
})();
