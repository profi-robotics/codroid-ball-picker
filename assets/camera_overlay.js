/**
 * Browser-side camera overlay renderer.
 *
 * The MJPEG image is intentionally raw. Overlay primitives are fetched as JSON
 * and drawn on a canvas so camera cadence is independent from overlay compute.
 */
(function initCameraOverlay() {
  const POLL_MS = 200;
  const HOLD_MS = 10000;
  let lastMeaningful = null;
  let lastStateKey = null;
  let stopped = false;

  function elements() {
    return {
      img: document.getElementById("camera-feed"),
      canvas: document.getElementById("camera-overlay-canvas"),
    };
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, rect };
  }

  function imageMapping(img, canvasRect, frameWidth, frameHeight) {
    const naturalWidth = frameWidth || img.naturalWidth || 0;
    const naturalHeight = frameHeight || img.naturalHeight || 0;
    if (!naturalWidth || !naturalHeight) {
      return null;
    }
    const scale = Math.min(
      canvasRect.width / naturalWidth,
      canvasRect.height / naturalHeight
    );
    const drawWidth = naturalWidth * scale;
    const drawHeight = naturalHeight * scale;
    return {
      scale,
      x: (canvasRect.width - drawWidth) / 2,
      y: (canvasRect.height - drawHeight) / 2,
    };
  }

  function point(mapping, x, y) {
    return {
      x: mapping.x + Number(x || 0) * mapping.scale,
      y: mapping.y + Number(y || 0) * mapping.scale,
    };
  }

  function drawLabel(ctx, text, x, y, color) {
    if (!text) {
      return;
    }
    ctx.save();
    ctx.font = "13px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.textBaseline = "bottom";
    const metrics = ctx.measureText(text);
    const pad = 5;
    const height = 20;
    const rect = ctx.canvas.getBoundingClientRect();
    const maxX = Math.max(0, rect.width - metrics.width - pad * 2);
    const maxY = Math.max(height, rect.height);
    const labelX = Math.min(Math.max(0, x), maxX);
    const labelY = Math.min(Math.max(height, y), maxY);
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.fillRect(labelX, labelY - height, metrics.width + pad * 2, height);
    ctx.fillStyle = color || "rgb(255,255,255)";
    ctx.fillText(text, labelX + pad, labelY - 5);
    ctx.restore();
  }

  function drawBox(ctx, primitive, mapping) {
    const a = point(mapping, primitive.x1, primitive.y1);
    const b = point(mapping, primitive.x2, primitive.y2);
    const color = primitive.color || "rgb(0,255,0)";
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    drawLabel(ctx, primitive.label, a.x, a.y, color);
    ctx.restore();
  }

  function drawPoint(ctx, primitive, mapping) {
    const p = point(mapping, primitive.x, primitive.y);
    const color = primitive.color || "rgb(255,255,255)";
    const radius = Math.max(1, Number(primitive.radius || 4) * mapping.scale);
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = Number(primitive.lineWidth || 2);
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    if (primitive.fill === false) {
      ctx.stroke();
    } else {
      ctx.fill();
    }
    drawLabel(
      ctx,
      primitive.label,
      p.x + Number(primitive.label_dx || 8),
      p.y + Number(primitive.label_dy || -8),
      color
    );
    ctx.restore();
  }

  function drawCircle(ctx, primitive, mapping) {
    const p = point(mapping, primitive.x, primitive.y);
    const color = primitive.color || "rgb(255,255,255)";
    const radius = Math.max(1, Number(primitive.radius || 4) * mapping.scale);
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = Number(primitive.lineWidth || 2);
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    if (primitive.fill) {
      ctx.fill();
    } else {
      ctx.stroke();
    }
    drawLabel(
      ctx,
      primitive.label,
      p.x + Number(primitive.label_dx || 8),
      p.y + Number(primitive.label_dy || -8),
      color
    );
    ctx.restore();
  }

  function drawCross(ctx, primitive, mapping) {
    const p = point(mapping, primitive.x, primitive.y);
    const color = primitive.color || "rgb(255,255,255)";
    const half = Math.max(3, Number(primitive.size || 12) * mapping.scale / 2);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Number(primitive.lineWidth || 2);
    ctx.beginPath();
    ctx.moveTo(p.x - half, p.y);
    ctx.lineTo(p.x + half, p.y);
    ctx.moveTo(p.x, p.y - half);
    ctx.lineTo(p.x, p.y + half);
    ctx.stroke();
    drawLabel(
      ctx,
      primitive.label,
      p.x + Number(primitive.label_dx || 8),
      p.y + Number(primitive.label_dy || -8),
      color
    );
    ctx.restore();
  }

  function drawLine(ctx, primitive, mapping) {
    const from = primitive.from || {};
    const to = primitive.to || {};
    const a = point(mapping, from.x, from.y);
    const b = point(mapping, to.x, to.y);
    const color = primitive.color || "rgb(255,255,255)";
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Number(primitive.lineWidth || 2);
    if (primitive.dashed) {
      ctx.setLineDash([6, 4]);
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    if (primitive.type === "arrow") {
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const len = 10;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x - len * Math.cos(angle - 0.5), b.y - len * Math.sin(angle - 0.5));
      ctx.lineTo(b.x - len * Math.cos(angle + 0.5), b.y - len * Math.sin(angle + 0.5));
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    }
    const labelAnchor = primitive.label_at === "start" ? a : b;
    drawLabel(
      ctx,
      primitive.label,
      labelAnchor.x + Number(primitive.label_dx || 6),
      labelAnchor.y + Number(primitive.label_dy || -6),
      color
    );
    ctx.restore();
  }

  function drawPolygon(ctx, primitive, mapping) {
    const points = primitive.points || [];
    if (points.length < 2) {
      return;
    }
    const color = primitive.color || "rgb(255,255,255)";
    ctx.save();
    ctx.globalAlpha = Number(primitive.alpha || 1);
    ctx.strokeStyle = color;
    ctx.fillStyle = primitive.fill || "rgba(0,0,0,0)";
    ctx.lineWidth = Number(primitive.lineWidth || 2);
    ctx.beginPath();
    points.forEach((raw, index) => {
      const p = Array.isArray(raw)
        ? point(mapping, raw[0], raw[1])
        : point(mapping, raw.x, raw.y);
      if (index === 0) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
    });
    ctx.closePath();
    if (primitive.fill) {
      ctx.fill();
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawPrimitive(ctx, primitive, mapping) {
    if (primitive.type === "box") {
      drawBox(ctx, primitive, mapping);
    } else if (primitive.type === "point") {
      drawPoint(ctx, primitive, mapping);
    } else if (primitive.type === "circle") {
      drawCircle(ctx, primitive, mapping);
    } else if (primitive.type === "cross") {
      drawCross(ctx, primitive, mapping);
    } else if (primitive.type === "line" || primitive.type === "arrow") {
      drawLine(ctx, primitive, mapping);
    } else if (primitive.type === "polygon") {
      drawPolygon(ctx, primitive, mapping);
    } else if (primitive.type === "text") {
      const p = point(mapping, primitive.x, primitive.y);
      drawLabel(ctx, primitive.text, p.x, p.y, primitive.color);
    }
  }

  function chooseState(state) {
    const stateKey = `${state.key || ""}:${state.generation || ""}`;
    if (stateKey !== lastStateKey) {
      lastStateKey = stateKey;
      lastMeaningful = null;
    }
    if (state.has_targets && Array.isArray(state.primitives) && state.primitives.length) {
      lastMeaningful = { state, receivedAt: Date.now() };
      return state;
    }
    if (
      lastMeaningful &&
      lastMeaningful.state.key === state.key &&
      Date.now() - lastMeaningful.receivedAt < HOLD_MS
    ) {
      return { ...lastMeaningful.state, stale: true };
    }
    return state;
  }

  function render(state) {
    const { img, canvas } = elements();
    if (!img || !canvas) {
      return;
    }
    const { ctx, rect } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!state || !state.key) {
      lastMeaningful = null;
      return;
    }
    const displayState = chooseState(state);
    const mapping = imageMapping(
      img,
      rect,
      displayState.frame_width,
      displayState.frame_height
    );
    if (!mapping) {
      return;
    }
    ctx.save();
    if (displayState.stale) {
      ctx.globalAlpha = 0.62;
      ctx.setLineDash([6, 4]);
    }
    (displayState.primitives || []).forEach((primitive) => {
      drawPrimitive(ctx, primitive, mapping);
    });
    ctx.restore();
  }

  async function poll() {
    if (stopped) {
      return;
    }
    const { img, canvas } = elements();
    if (!img || !canvas) {
      lastMeaningful = null;
      setTimeout(poll, POLL_MS);
      return;
    }
    try {
      const response = await fetch(`/overlay_state.json?ts=${Date.now()}`, {
        cache: "no-store",
      });
      if (response.ok) {
        render(await response.json());
      }
    } catch (_err) {
      // Keep the last drawn overlay during transient polling failures.
    } finally {
      setTimeout(poll, POLL_MS);
    }
  }

  window.addEventListener("resize", () => {
    fetch(`/overlay_state.json?ts=${Date.now()}`, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (state) {
          render(state);
        }
      })
      .catch(() => {});
  });

  document.addEventListener("visibilitychange", () => {
    stopped = document.hidden;
    if (!stopped) {
      poll();
    }
  });

  poll();
})();
