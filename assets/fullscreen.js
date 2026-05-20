/**
 * Delegated fullscreen handler for the camera feed.
 *
 * The camera feed node is replaced when camera connection state changes, so bind
 * to the document once instead of polling for a specific element at startup.
 */
(function initFullscreen() {
  function cameraFeed() {
    return (
      document.getElementById("camera-layered-feed") ||
      document.getElementById("camera-feed")
    );
  }

  function requestFullscreen() {
    const feed = cameraFeed();
    if (!feed) {
      return;
    }

    const fsFn =
      feed.requestFullscreen ||
      feed.webkitRequestFullscreen ||
      feed.mozRequestFullScreen ||
      feed.msRequestFullscreen;

    if (fsFn) {
      const result = fsFn.call(feed);
      if (result && typeof result.catch === "function") {
        result.catch(() => {
          if (feed.webkitEnterFullscreen) {
            feed.webkitEnterFullscreen();
          }
        });
      }
      return;
    }

    if (feed.webkitEnterFullscreen) {
      feed.webkitEnterFullscreen();
    }
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element) || !target.closest("#fullscreen-btn")) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    requestFullscreen();
  });

  document.addEventListener("dblclick", (event) => {
    const target = event.target;
    if (
      !(target instanceof Element) ||
      !target.closest("#camera-layered-feed, #camera-feed")
    ) {
      return;
    }
    event.preventDefault();
    requestFullscreen();
  });
})();
