(function () {
  function setOverlay(mode) {
    fetch("/camera/overlay", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode})
    }).catch(() => {});
    const img = document.getElementById("camera-stream");
    if (img) {
      img.src = `/video/stream?overlay=${encodeURIComponent(mode)}&t=${Date.now()}`;
    }
  }

  function fallbackSnapshot() {
    const img = document.getElementById("camera-stream");
    if (!img) return;
    img.src = `/video/snapshot.jpg?t=${Date.now()}`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-overlay]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-overlay]").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        setOverlay(button.dataset.overlay);
      });
    });
    const fullscreen = document.getElementById("fullscreen-video");
    const img = document.getElementById("camera-stream");
    if (fullscreen && img) {
      fullscreen.addEventListener("click", () => {
        if (img.requestFullscreen) img.requestFullscreen();
      });
      img.addEventListener("error", fallbackSnapshot);
    }
  });
})();

