(function initSidebarResize() {
  const storageKey = "codroid-picker-sidebar-width";
  const minWidth = 280;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function maxWidthFor(grid) {
    return Math.max(minWidth, Math.min(620, Math.floor(grid.clientWidth * 0.58)));
  }

  function setSidebarWidth(width) {
    document.documentElement.style.setProperty("--sidebar-width", `${width}px`);
  }

  function savedSidebarWidth() {
    try {
      return Number(window.localStorage.getItem(storageKey));
    } catch (error) {
      return Number.NaN;
    }
  }

  function saveSidebarWidth(width) {
    try {
      window.localStorage.setItem(storageKey, String(width));
    } catch (error) {
      // Ignore storage failures; dragging should still work for this page load.
    }
  }

  function restoreSidebarWidth() {
    const savedWidth = savedSidebarWidth();
    const grid = document.querySelector(".layout-grid > .mantine-Grid-inner");
    if (!grid || !Number.isFinite(savedWidth)) {
      return;
    }
    setSidebarWidth(clamp(savedWidth, minWidth, maxWidthFor(grid)));
  }

  function initHandle(handle) {
    if (handle.dataset.resizeReady === "true") {
      return;
    }
    handle.dataset.resizeReady = "true";

    handle.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 47.99em)").matches) {
        return;
      }

      const grid = handle.closest(".layout-grid")?.querySelector(".mantine-Grid-inner");
      if (!grid) {
        return;
      }

      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      document.body.classList.add("sidebar-resizing");

      const onPointerMove = (moveEvent) => {
        const gridRect = grid.getBoundingClientRect();
        const width = moveEvent.clientX - gridRect.left;
        const nextWidth = clamp(Math.round(width), minWidth, maxWidthFor(grid));
        setSidebarWidth(nextWidth);
        saveSidebarWidth(nextWidth);
      };

      const stopResize = () => {
        document.body.classList.remove("sidebar-resizing");
        handle.removeEventListener("pointermove", onPointerMove);
        handle.removeEventListener("pointerup", stopResize);
        handle.removeEventListener("pointercancel", stopResize);
      };

      handle.addEventListener("pointermove", onPointerMove);
      handle.addEventListener("pointerup", stopResize);
      handle.addEventListener("pointercancel", stopResize);
    });
  }

  function init() {
    restoreSidebarWidth();
    const handle = document.getElementById("sidebar-resizer");
    if (handle) {
      initHandle(handle);
      return true;
    }
    return false;
  }

  function observeHandle() {
    const observer = new MutationObserver(() => {
      if (init()) {
        observer.disconnect();
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }

  window.addEventListener("resize", restoreSidebarWidth);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      if (!init()) {
        observeHandle();
      }
    });
  } else if (!init()) {
    observeHandle();
  }
})();
