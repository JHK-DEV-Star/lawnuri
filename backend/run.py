"""
Backend entry point for the LawNuri server.

Usage:
    python run.py

Environment variables:
    UVICORN_PORT  - Server port (default: 8000)
    DEBUG         - Enable debug/reload mode (default: True)
"""

import os
import sys

# In PyInstaller windowed (console=False) bundles, sys.stdout/sys.stderr are None.
# uvicorn's ColourizedFormatter calls sys.stdout.isatty() during init, which crashes.
# Replace None streams with /dev/null so isatty() returns False and color is disabled.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Add project root (backend/) to sys.path so 'app' package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _start_parent_watchdog():
    """Exit this process when the parent (Flutter UI) terminates.

    Frontend passes its PID via LAWNURI_PARENT_PID. The backend waits on the
    parent handle in a daemon thread and self-exits when the parent dies —
    handles graceful close, hard kill, crash, etc. uniformly.
    """
    parent_pid_str = os.environ.get("LAWNURI_PARENT_PID")
    if not parent_pid_str:
        return
    try:
        parent_pid = int(parent_pid_str)
    except ValueError:
        return

    import threading

    if sys.platform == "win32":
        import ctypes

        PROCESS_SYNCHRONIZE = 0x00100000
        INFINITE = 0xFFFFFFFF
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_SYNCHRONIZE, False, parent_pid)
        if not handle:
            return

        def _waiter():
            kernel32.WaitForSingleObject(handle, INFINITE)
            os._exit(0)

        threading.Thread(target=_waiter, daemon=True).start()
    else:
        import time

        def _poller():
            while True:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    os._exit(0)
                time.sleep(2)

        threading.Thread(target=_poller, daemon=True).start()


def main():
    """Run the uvicorn ASGI server."""
    import uvicorn

    _start_parent_watchdog()

    port = int(os.getenv("UVICORN_PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    # uvicorn's reload mode spawns a watched subprocess, which doesn't work in
    # PyInstaller frozen bundles. Force-disable reload when frozen.
    if getattr(sys, "frozen", False):
        debug = False

    print(f"Starting LawNuri backend on http://0.0.0.0:{port}")
    print(f"Debug/reload mode: {debug}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
