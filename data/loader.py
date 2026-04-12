"""
DataLoader — loads pre-generated episode batches from disk.

Used by GSTEnvironment.reset() when split != None, so that training,
validation, and test runs use the same fixed episode data.

Usage:
    loader = DataLoader(split="train")
    batch  = loader.get_batch(task_id=1, episode_idx=0)   # specific episode
    batch  = loader.get_batch(task_id=2)                  # random from split
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from models.observation import InvoiceObservation, ITCMismatch

DATA_DIR = Path(__file__).parent


class DataLoader:
    """
    Loads pre-generated episode JSON files from data/{split}/{task_id}/.

    Falls back to live generation (SyntheticDataGenerator) if the split
    directory doesn't exist yet (e.g. first run before generate_splits.py).
    """

    def __init__(self, split: str = "train", seed: Optional[int] = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self._rng  = random.Random(seed)
        self._index: dict[int, list[Path]] = {}
        self._build_index()

    # ──────────────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Index all episode files per task_id."""
        for task_id in (1, 2, 3):
            task_dir = DATA_DIR / self.split / str(task_id)
            if task_dir.exists():
                files = sorted(task_dir.glob("episode_*.json"))
                self._index[task_id] = files
            else:
                self._index[task_id] = []

    def available(self, task_id: int) -> int:
        """Number of pre-generated episodes for this task."""
        return len(self._index.get(task_id, []))

    def get_batch(
        self,
        task_id: int,
        episode_idx: Optional[int] = None,
    ) -> dict:
        """
        Load a batch dict from disk.

        If episode_idx is None, pick randomly within the split.
        Falls back to live generation if no files exist.
        """
        files = self._index.get(task_id, [])
        if not files:
            return self._live_generate(task_id)

        if episode_idx is not None:
            idx = episode_idx % len(files)
        else:
            idx = self._rng.randint(0, len(files) - 1)

        with open(files[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        return self._deserialise_batch(record["batch"], task_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Deserialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _deserialise_batch(self, batch: dict, task_id: int) -> dict:
        """Reconstruct Pydantic objects from plain dicts."""
        out = dict(batch)

        if "invoices" in out:
            out["invoices"] = [
                InvoiceObservation(**inv) if isinstance(inv, dict) else inv
                for inv in out["invoices"]
            ]

        if "mismatches" in out:
            out["mismatches"] = [
                ITCMismatch(**m) if isinstance(m, dict) else m
                for m in out["mismatches"]
            ]

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Live fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _live_generate(self, task_id: int) -> dict:
        """Generate on-the-fly when disk data is not available."""
        from data.synthetic.generator import SyntheticDataGenerator
        seed = self._rng.randint(0, 2**31 - 1)
        gen  = SyntheticDataGenerator(seed=seed)
        if task_id == 1:
            return gen.generate_task1_batch()
        elif task_id == 2:
            return gen.generate_task2_batch()
        else:
            return gen.generate_task3_batch()
