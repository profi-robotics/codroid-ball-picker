/**
 * Fullscreen handler for video element
 * Executes immediately on load, no waiting for DOMContentLoaded
 */
(function initFullscreen() {
  let attempts = 0;
  const maxAttempts = 50;

  function bindHandlers() {
    const fullscreenBtn = document.getElementById("fullscreen-btn");
    const cameraFeed = document.getElementById("camera-feed");

    attempts++;

    if (!fullscreenBtn || !cameraFeed) {
      if (attempts < maxAttempts) {
        setTimeout(bindHandlers, 100);
      }
      return;
    }

    function requestFullscreen() {
      const fsFn =
        cameraFeed.requestFullscreen ||
        cameraFeed.webkitRequestFullscreen ||
        cameraFeed.mozRequestFullScreen ||
        cameraFeed.msRequestFullscreen;

      if (fsFn) {
        fsFn.call(cameraFeed).catch((err) => {
          if (cameraFeed.webkitEnterFullscreen) {
            cameraFeed.webkitEnterFullscreen();
          }
        });
      } else if (cameraFeed.webkitEnterFullscreen) {
        cameraFeed.webkitEnterFullscreen();
      }
    }

    fullscreenBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      requestFullscreen();
    });

    cameraFeed.addEventListener("dblclick", (e) => {
      e.preventDefault();
      requestFullscreen();
    });
  }

  // Try immediately in case DOM is ready
  bindHandlers();
})();
