from __future__ import annotations

import uuid
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from models.observation import GSTObservation, GSTR3BSummary
from models.action import GSTAction, ActionType
from models.reward import RewardSignal
from env.rewards import RewardEngine
from env.memory_manager import MemoryManager
from env.tasks.task1_classify import Task1Classify
from env.tasks.task2_itc import Task2ITC
from env.tasks.task3_filing import Task3Filing


# Valid actions per environment phase
VALID_ACTIONS: dict[str, list[str]] = {
    "CLASSIFYING": [
        ActionType.CLASSIFY_INVOICE,
        ActionType.SKIP_INVOICE,
        ActionType.REQUEST_CLARIFICATION,
    ],
    "RECONCILING": [
        ActionType.MATCH_ITC,
        ActionType.FLAG_DISCREPANCY,
        ActionType.ACCEPT_MISMATCH,
        ActionType.DEFER_INVOICE,
        ActionType.SKIP_INVOICE,
    ],
    "FILING": [
        ActionType.SET_SECTION_VALUE,
        ActionType.GENERATE_RETURN,
        ActionType.SUBMIT_RETURN,
    ],
}

MAX_STEPS_PER_TASK = {1: 50, 2: 150, 3: 300}
TASK_PHASE = {1: "CLASSIFYING", 2: "RECONCILING", 3: "FILING"}


class GSTEnvironment(gym.Env):
    """
    OpenEnv-compliant GST workflow environment.

    Tasks:
        1 — invoice_classification  (easy,   max 50  steps)
        2 — itc_reconciliation      (medium, max 150 steps)
        3 — gstr3b_filing           (hard,   max 300 steps)

    Passes: openenv validate
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: Optional[dict] = None, split: Optional[str] = None):
        super().__init__()
        self.config        = config or {}
        self.split         = split   # "train" | "val" | "test" | None (live)
        self.reward_engine = RewardEngine()
        self.memory        = MemoryManager()
        self._state: dict  = {}
        self._task_handler = None   # set in reset()
        self._loader       = None   # set lazily on first reset if split is set

        # Gymnasium spaces — flattened dict representation
        # (kept minimal; real structure is in Pydantic models)
        self.observation_space = gym.spaces.Dict({
            "task_id":          gym.spaces.Discrete(4),          # 1-3
            "step_number":      gym.spaces.Box(0, 1000, shape=(1,), dtype=np.int32),
            "pending_count":    gym.spaces.Box(0, 1000, shape=(1,), dtype=np.int32),
            "cumulative_reward": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(len(ActionType))

    # ──────────────────────────────────────────────────────────────────────────
    # OpenEnv interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        task_id: int = 1,
        episode_idx: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[GSTObservation, dict]:
        """
        Load a new invoice batch. Returns initial GSTObservation.

        Args:
            seed:        RNG seed (used for live generation when split=None)
            task_id:     1 (classify) | 2 (ITC) | 3 (filing)
            episode_idx: if split is set, load this specific episode index;
                         None = random episode from the split
        """
        super().reset(seed=seed)
        self.memory.reset()

        self._task_id   = task_id
        self._max_steps = MAX_STEPS_PER_TASK.get(task_id, 50)
        self._episode_id = f"ep-{uuid.uuid4().hex[:12]}"

        # Initialise task handler (used only for apply_action / is_done)
        if task_id == 1:
            self._task_handler = Task1Classify(seed=seed)
        elif task_id == 2:
            self._task_handler = Task2ITC(seed=seed)
        elif task_id == 3:
            self._task_handler = Task3Filing(seed=seed)
        else:
            raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")

        # Load batch: from disk (split) or live generation
        if self.split is not None:
            if self._loader is None or self._loader.split != self.split:
                from data.loader import DataLoader
                self._loader = DataLoader(split=self.split, seed=seed)
            batch = self._loader.get_batch(task_id=task_id, episode_idx=episode_idx)
        else:
            batch = self._task_handler.get_batch()

        self._state = {
            # Episode metadata
            "episode_id":            self._episode_id,
            "task_id":               task_id,
            "phase":                 TASK_PHASE[task_id],
            "step":                  0,
            "max_steps":             self._max_steps,
            "gstin":                 batch.get("gstin", "27AABCU9603R1ZM"),
            "tax_period":            batch.get("tax_period", "2024-03"),

            # Invoice state
            "invoices":              batch.get("invoices", []),
            "current_invoice_idx":   0,
            "classified_count":      0,
            # Task 1: pending = invoice count; Task 2: pending = mismatch count; Task 3: 0
            "pending_count":         (
                len(batch.get("mismatches", [])) if task_id == 2
                else len(batch.get("invoices", []))
            ),
            "flagged_count":         0,

            # ITC state
            "mismatches":            batch.get("mismatches", []),
            "matched_itc":           0.0,
            "disputed_itc":          0.0,
            "eligible_itc":          batch.get("eligible_itc", 0.0),
            "purchase_map":          batch.get("purchase_map", {}),
            "gstr2b_map":            batch.get("gstr2b_map", {}),
            "best_match":            batch.get("best_match", {}),
            "correct_matches":       batch.get("correct_matches", {}),   # purchase_id → gstr2b_id
            "discrepancy_truth":     batch.get("discrepancy_truth", {}),
            "total_itc_claimed":     0.0,

            # Filing state
            "section_values":        {},
            "true_section_values":   batch.get("true_section_values", {}),
            "itc_offset_correct":    False,
            "unresolved_discrepancies": 0,

            # Reward tracking
            "cumulative_reward":     0.0,
            "correct_decisions":     0,
            "total_decisions":       0,
            "invalid_actions":       0,
            "seen_actions":          {},

            # Ground truth for graders
            "ground_truth":          batch.get("ground_truth", {}),
            "steps_remaining":       self._max_steps,
            "last_action_result":    None,
            "last_action_error":     None,
        }

        # Build supplier profiles into memory
        for gstin, profile in batch.get("supplier_profiles", {}).items():
            self.memory.update_supplier_profile(
                gstin,
                profile.get("compliance_rate", 1.0),
                profile.get("avg_delay_days", 0.0),
                profile.get("cancelled_pct", 0.0),
            )

        obs = self._build_observation()
        return obs, {}

    def step(self, action: GSTAction) -> tuple[GSTObservation, float, bool, bool, dict]:
        """
        Execute one agent action.
        Returns: (observation, reward, terminated, truncated, info)
        info["last_action_error"] is used for the [STEP] error field.
        """
        # Phase-validity check
        if not self._is_valid_action(action):
            reward = -0.15
            error_msg = (
                f"invalid_action_{action.action_type.value}_in_{self._state['phase']}"
            )
            self._state["step"] += 1
            self._state["steps_remaining"] = max(0, self._max_steps - self._state["step"])
            self._state["invalid_actions"] += 1
            self._state["last_action_result"] = "invalid"
            self._state["last_action_error"]  = error_msg
            self._state["cumulative_reward"]  += reward
            info = {
                "last_action_error": error_msg,
                "step":  self._state["step"],
                "phase": self._state["phase"],
                "sub_rewards": {"invalid_action": reward},
            }
            return self._build_observation(), reward, False, False, info

        # Apply action to task handler (mutates _state)
        self._task_handler.apply_action(action, self._state, self.memory)

        # Compute reward
        reward_signal: RewardSignal = self.reward_engine.compute(action, self._state)

        self._state["step"] += 1
        self._state["steps_remaining"] = max(0, self._max_steps - self._state["step"])
        self._state["cumulative_reward"] = reward_signal.cumulative_reward
        self._state["last_action_result"] = (
            "penalty" if reward_signal.penalty_flags else "success"
        )
        self._state["last_action_error"] = None

        terminated = self._check_terminal()
        truncated  = self._state["step"] >= self._max_steps

        info = {
            "last_action_error": None,
            "step":              self._state["step"],
            "phase":             self._state["phase"],
            "sub_rewards":       reward_signal.sub_rewards,
            "penalty_flags":     reward_signal.penalty_flags,
        }
        return self._build_observation(), reward_signal.step_reward, terminated, truncated, info

    def state(self) -> dict:
        """Full internal state for deterministic replay and audit."""
        return self._state.copy()

    def close(self):
        """Cleanup. [END] line must be printed AFTER this call in inference.py."""
        self._state.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _is_valid_action(self, action: GSTAction) -> bool:
        phase = self._state.get("phase", "CLASSIFYING")
        return action.action_type in VALID_ACTIONS.get(phase, [])

    def _check_terminal(self) -> bool:
        if self._task_handler:
            return self._task_handler.is_done(self._state)
        return False

    def _build_observation(self) -> GSTObservation:
        s = self._state
        invoices = s.get("invoices", [])
        idx      = s.get("current_invoice_idx", 0)
        current  = invoices[idx] if idx < len(invoices) else None

        # Retrieve memory context
        hsn = getattr(current, "hsn_code", None) if current else None
        gstin = getattr(current, "supplier_gstin", None) if current else None

        return GSTObservation(
            episode_id=s.get("episode_id", ""),
            task_id=s.get("task_id", 1),
            step_number=s.get("step", 0),
            gstin=s.get("gstin", ""),
            tax_period=s.get("tax_period", ""),
            current_invoice=current,
            total_invoices=len(invoices),
            classified_count=s.get("classified_count", 0),
            pending_count=s.get("pending_count", 0),
            flagged_count=s.get("flagged_count", 0),
            mismatches=s.get("mismatches", []),
            matched_itc_amount=s.get("matched_itc", 0.0),
            disputed_itc_amount=s.get("disputed_itc", 0.0),
            gstr3b=GSTR3BSummary(**s.get("gstr3b_summary", {})) if s.get("gstr3b_summary") else GSTR3BSummary(),
            steps_remaining=s.get("steps_remaining", self._max_steps),
            cumulative_reward=s.get("cumulative_reward", 0.0),
            last_action_result=s.get("last_action_result"),
            similar_past_invoices=self.memory.get_similar_invoices(hsn),
            known_supplier_profile=self.memory.get_supplier_profile(gstin),
        )
