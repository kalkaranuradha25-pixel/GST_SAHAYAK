"""
Pre-generate train / val / test episode splits and save to disk.

Usage:
    python data/generate_splits.py

Output:
    data/train/  — 800 episodes (seed 42,  tasks 1/2/3 distributed)
    data/val/    — 100 episodes (seed 999, same distribution)
    data/test/   — 100 episodes (manually curated seeds for edge-case coverage)

Each episode is saved as:
    {split}/{task_id}/episode_{idx:04d}.json

Format per file:
    {
        "episode_idx": int,
        "task_id":     int,
        "seed":        int,
        "split":       "train" | "val" | "test",
        "batch":       { ...task batch dict... }
    }

Note: InvoiceObservation objects are serialised via .model_dump().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root or from data/
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic.generator import SyntheticDataGenerator
from models.observation import InvoiceObservation, ITCMismatch

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SPLITS: dict[str, dict] = {
    "train": {"n": 800, "base_seed": 42},
    "val":   {"n": 100, "base_seed": 999},
    "test":  {"n": 100, "base_seed": None},   # curated seeds below
}

# Task distribution per episode index (round-robin): 1→easy, 2→medium, 3→hard
def task_for_idx(idx: int) -> int:
    return (idx % 3) + 1   # 0→1, 1→2, 2→3, 3→1, ...

# Test split uses hand-picked seeds to guarantee edge-case coverage
# 34 task-1 + 33 task-2 + 33 task-3 = 100
TEST_SEEDS: list[tuple[int, int]] = (
    [(1, 1000 + i) for i in range(34)] +
    [(2, 2000 + i) for i in range(33)] +
    [(3, 3000 + i) for i in range(33)]
)

DATA_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_batch(batch: dict) -> dict:
    """Convert Pydantic objects inside a batch dict to plain dicts."""
    out = {}
    for key, val in batch.items():
        if isinstance(val, list):
            out[key] = [_serialise_item(v) for v in val]
        elif isinstance(val, dict):
            out[key] = {k: _serialise_item(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def _serialise_item(item: Any) -> Any:
    if isinstance(item, (InvoiceObservation, ITCMismatch)):
        return item.model_dump()
    if isinstance(item, dict):
        return {k: _serialise_item(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_serialise_item(v) for v in item]
    return item


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_episode(task_id: int, seed: int) -> dict:
    gen = SyntheticDataGenerator(seed=seed)
    if task_id == 1:
        batch = gen.generate_task1_batch()
    elif task_id == 2:
        batch = gen.generate_task2_batch()
    else:
        batch = gen.generate_task3_batch()
    return _serialise_batch(batch)


def write_episode(split: str, task_id: int, idx: int, seed: int, batch: dict):
    out_dir = DATA_DIR / split / str(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_idx": idx,
        "task_id":     task_id,
        "seed":        seed,
        "split":       split,
        "batch":       batch,
    }
    path = out_dir / f"episode_{idx:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, separators=(",", ":"))


def generate_split(split: str, n: int, base_seed: int | None):
    print(f"Generating {split} ({n} episodes) ...")
    counts = {1: 0, 2: 0, 3: 0}

    if split == "test":
        items = TEST_SEEDS[:n]
        for idx, (task_id, seed) in enumerate(items):
            batch = generate_episode(task_id, seed)
            write_episode(split, task_id, idx, seed, batch)
            counts[task_id] += 1
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{n}")
    else:
        for idx in range(n):
            task_id = task_for_idx(idx)
            seed    = base_seed + idx
            batch   = generate_episode(task_id, seed)
            write_episode(split, task_id, idx, seed, batch)
            counts[task_id] += 1
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{n}")

    print(f"  Done. task1={counts[1]} task2={counts[2]} task3={counts[3]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split_name, cfg in SPLITS.items():
        generate_split(split_name, cfg["n"], cfg["base_seed"])

    # Write a manifest file for quick inspection
    manifest = {}
    for split_name, cfg in SPLITS.items():
        manifest[split_name] = {
            "n_episodes": cfg["n"],
            "base_seed":  cfg["base_seed"],
            "task_distribution": "round-robin 1/2/3" if split_name != "test" else "34/33/33 curated",
        }
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("manifest.json written")
    print("All splits generated.")
