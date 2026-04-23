"""
server/app.py — OpenEnv multi-mode deployment entry point.

Re-exports the FastAPI `app` from the root server.py and adds a
`main()` function so the `server` CLI script in pyproject.toml works.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT_SERVER_PATH = Path(__file__).resolve().parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("gst_root_server", ROOT_SERVER_PATH)
_gst_root_server = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_gst_root_server)
app = _gst_root_server.app


def main():
    """Entry point for `server` CLI defined in pyproject.toml."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
