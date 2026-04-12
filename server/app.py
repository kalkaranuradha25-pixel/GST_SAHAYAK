"""
server/app.py — OpenEnv multi-mode deployment entry point.

Re-exports the FastAPI `app` from the root server.py and adds a
`main()` function so the `server` CLI script in pyproject.toml works.
"""
from server import app  # noqa: F401  (root server.py, not this package)


def main():
    """Entry point for `server` CLI defined in pyproject.toml."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
