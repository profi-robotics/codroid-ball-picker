"""Run the Dash calibration UI."""

from __future__ import annotations

import argparse
import atexit
import logging
import sys
import os
import time
import importlib.util


def is_reloader_process() -> bool:
    """Check if this is the reloader parent process (not the worker)."""
    # In debug mode, Werkzeug sets WERKZEUG_RUN_MAIN='true' in the worker process.
    # The parent (reloader) process does NOT have this set.
    return os.environ.get('WERKZEUG_RUN_MAIN') != 'true'


def cleanup_on_reload():
    """Cleanup handler called on app shutdown or reload."""
    logger = logging.getLogger(__name__)
    logger.info("App shutdown detected - starting cleanup...")

    try:
        from app.codroid_picker.programs import stop_program, get_registry
        from app.codroid_picker.streaming import stop_camera_stream
        from app.settings import _save_settings, SETTINGS

        # Save settings
        try:
            _save_settings(SETTINGS)
            logger.info("Settings saved")
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)

        # Stop active programs
        try:
            registry = get_registry()
            if registry.is_program_running:
                stop_program()
                logger.info("Program stopped")
        except Exception as exc:
            logger.error("Failed to stop program: %s", exc)

        # Close camera stream with delay for USB device release
        try:
            stop_camera_stream(wait_for_release=True)
            logger.info("Camera stream closed")
        except Exception as exc:
            logger.error("Failed to close camera stream: %s", exc)

        # Close robot session
        try:
            from app.codroid_picker.app_instance import AppRegistry
            app_registry = AppRegistry()
            if app_registry.is_initialized and app_registry._robot_session is not None:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(app_registry._robot_session.close())
                    else:
                        loop.run_until_complete(
                            app_registry._robot_session.close())
                except Exception:
                    pass
                logger.info("Robot session closed")
        except Exception as exc:
            logger.error("Failed to close robot session: %s", exc)

        # Close gripper connection
        try:
            from app.codroid_picker.session import SESSION
            if SESSION.gripper is not None:
                SESSION.gripper = None
                SESSION.gripper_device = None
                logger.info("Gripper connection cleared")
        except Exception as exc:
            logger.error("Failed to close gripper: %s", exc)

        # Reset singletons for clean reload
        try:
            from app.codroid_picker.app_instance import AppRegistry
            from app.codroid_picker.programs.registry import ProgramRegistry
            AppRegistry.reset()
            ProgramRegistry.reset()
            logger.info("Singletons reset")
        except Exception as exc:
            logger.error("Failed to reset singletons: %s", exc)

        # Reset initialization flag for main module
        try:
            if "codroid_picker_main" in sys.modules:
                codroid_main = sys.modules["codroid_picker_main"]
                if hasattr(codroid_main, "reset_initialization"):
                    codroid_main.reset_initialization()
                    logger.info("Main module initialization reset")
        except Exception as exc:
            logger.error("Failed to reset main initialization: %s", exc)

        logger.info("Cleanup completed")

    except Exception as exc:
        logger.error("Error during cleanup: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Codroid calibration UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode.")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('asyncio').setLevel(logging.INFO)
    logging.getLogger('websockets').setLevel(logging.INFO)

    # In debug mode with reloader, only the worker process should initialize
    # the app module and register cleanup handlers.
    if args.debug and is_reloader_process():
        # This is the reloader parent process - just start the reloader
        # without loading the full app module (which has side effects)
        logging.info("Reloader process starting - will spawn worker...")

        # We still need to create a minimal app for the reloader to work
        from dash import Dash
        app = Dash(__name__, suppress_callback_exceptions=True)

        import glob
        extra_files = glob.glob('app/**/*.py', recursive=True)
        extra_files += glob.glob('vision/**/*.py', recursive=True)

        app.run(
            host=args.host,
            port=args.port,
            debug=True,
            extra_files=extra_files,
        )
        return

    # Worker process (or non-debug mode) - do full initialization
    logging.info("Worker process starting - initializing app...")

    # Load the main app module
    spec = importlib.util.spec_from_file_location(
        "codroid_picker_main", "app/main.py")
    codroid_picker = importlib.util.module_from_spec(spec)
    sys.modules["codroid_picker_main"] = codroid_picker
    spec.loader.exec_module(codroid_picker)

    # Register cleanup handler
    atexit.register(cleanup_on_reload)

    # Collect all Python files to watch for changes
    import glob
    extra_files = glob.glob('app/**/*.py', recursive=True)
    extra_files += glob.glob('vision/**/*.py', recursive=True)

    # Create and run the app
    app = codroid_picker.create_app()
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        extra_files=extra_files if args.debug else None,
    )


if __name__ == "__main__":
    main()
