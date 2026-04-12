"""
server.py — FastAPI server exposing the GST environment as an HTTP API.

Endpoints (required by openenv validate):
    POST /reset          → GSTObservation (JSON)
    POST /step           → {observation, reward, terminated, truncated, info}
    GET  /state          → current state dict
    POST /close          → {}

Extra endpoints:
    GET  /health         → {"status": "ok"}
    GET  /tasks          → list of available tasks
    POST /run_episode    → run a full inference episode (calls inference.py logic)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.gst_env import GSTEnvironment
from models.observation import GSTObservation
from models.action import GSTAction

app = FastAPI(
    title="GST Intelligence RL Environment",
    description="OpenEnv-compliant GST workflow environment for hackathon evaluation.",
    version="1.0.0",
)

# One shared environment instance per server process
_env: Optional[GSTEnvironment] = None


def _get_env() -> GSTEnvironment:
    global _env
    if _env is None:
        _env = GSTEnvironment()
    return _env


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:     int            = 1
    seed:        Optional[int]  = 42
    episode_idx: Optional[int]  = None
    split:       Optional[str]  = None   # "train"|"val"|"test"|None


class StepResponse(BaseModel):
    observation: dict
    reward:      float
    terminated:  bool
    truncated:   bool
    info:        dict


class EpisodeRequest(BaseModel):
    task_id: int           = 1
    seed:    Optional[int] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "gst-intelligence-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": 1, "name": "invoice_classification", "difficulty": "easy",   "max_steps": 50},
            {"id": 2, "name": "itc_reconciliation",     "difficulty": "medium", "max_steps": 150},
            {"id": 3, "name": "gstr3b_filing",          "difficulty": "hard",   "max_steps": 300},
        ]
    }


@app.post("/reset", response_model=dict)
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = GSTEnvironment(split=req.split)
    obs, info = _env.reset(
        seed=req.seed,
        task_id=req.task_id,
        episode_idx=req.episode_idx,
    )
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(action: GSTAction):
    env = _get_env()
    if not env._state:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    obs, reward, terminated, truncated, info = env.step(action)
    # Serialise info (may contain non-JSON-serialisable values)
    safe_info = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                 for k, v in info.items()}
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=safe_info,
    )


@app.get("/state")
def state():
    env = _get_env()
    s = env.state()
    # Remove non-serialisable Pydantic objects before returning
    safe = {}
    for k, v in s.items():
        if isinstance(v, list):
            safe[k] = [i.model_dump() if hasattr(i, "model_dump") else i for i in v]
        elif hasattr(v, "model_dump"):
            safe[k] = v.model_dump()
        elif isinstance(v, (str, int, float, bool, dict, type(None))):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


@app.post("/close")
def close():
    env = _get_env()
    env.close()
    return {}


@app.post("/run_episode")
def run_episode_endpoint(req: EpisodeRequest = EpisodeRequest()):
    """
    Run a full episode via the LLM inference agent.
    Returns the [START]/[STEP]/[END] log lines as a list.
    Requires HF_TOKEN environment variable to be set.
    """
    import io
    import contextlib
    import sys

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=400, detail="HF_TOKEN not set")

    # Capture stdout from inference.py's run_episode()
    from inference import run_episode
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        success = run_episode(task_id=req.task_id, seed=req.seed or 42)

    lines = buffer.getvalue().strip().split("\n")
    return {"success": success, "log": lines}
