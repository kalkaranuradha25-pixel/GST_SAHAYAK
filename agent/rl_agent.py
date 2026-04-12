"""
rl_agent.py — PPO-based RL Agent (stable-baselines3)

Wraps GSTEnvironment in a flat-observation gym.Env compatible with SB3,
then trains a PPO policy across all 3 tasks.

Usage — Training:
    python agent/rl_agent.py --train --timesteps 500000

Usage — Evaluation:
    python agent/rl_agent.py --eval --model checkpoints/ppo_gst_final

Usage — Loading a saved policy:
    from agent.rl_agent import RLAgent
    agent = RLAgent.load("checkpoints/ppo_gst_final")
    obs, _ = env.reset(task_id=1)
    action = agent.act(obs, task_id=1)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Lazy SB3 imports (not required at inference time)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, BaseCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from models.observation import GSTObservation
from models.action import (
    GSTAction, ActionType,
    ClassifyInvoicePayload, MatchITCPayload,
    FlagDiscrepancyPayload, AcceptMismatchPayload,
    DeferInvoicePayload, SetSectionValuePayload,
    GenerateReturnPayload, SubmitReturnPayload,
    SkipInvoicePayload,
)

CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"
GSTR3B_SECTION_ORDER = [
    "3.1a", "3.1b", "3.1c", "3.1d",
    "4a", "4b", "6.1", "6.2", "6.3",
]


# ─────────────────────────────────────────────────────────────────────────────
# Flat observation wrapper for SB3
# ─────────────────────────────────────────────────────────────────────────────

class FlatGSTEnv(gym.Env):
    """
    Flattens GSTObservation into a fixed-size float32 vector for SB3.

    Observation vector (32 dims):
        [0]   task_id (1-3, normalised to 0-1)
        [1]   step_number / max_steps
        [2]   pending_count / 50
        [3]   classified_count / 50
        [4]   flagged_count / 50
        [5]   cumulative_reward / 10
        [6]   steps_remaining / max_steps
        [7]   matched_itc / 1e6
        [8]   disputed_itc / 1e6
        [9]   taxable_outward / 1e7
        [10]  itc_igst / 1e6
        [11]  itc_ineligible / 1e6
        [12]  igst_payable / 1e6
        [13]  cgst_payable / 1e6
        [14]  sgst_payable / 1e6
        [15]  net_payable / 1e6
        [16]  sections_completed_count / 9
        [17]  current_inv_taxable / 1e6
        [18]  current_inv_igst / 1e5
        [19]  current_inv_slab (0/5/12/18/28 → 0-1)
        [20]  current_inv_type (0-5)
        [21]  current_inv_has_flags (0/1)
        [22]  current_inv_fake (0/1)
        [23]  current_inv_missing_gstin (0/1)
        [24]  top_mismatch_delta / 1e5
        [25]  top_mismatch_type (0-5)
        [26]  supplier_compliance (0-1, 0.5 if unknown)
        [27]  last_reward_positive (0/1)
        [28]  phase_classifying (0/1)
        [29]  phase_reconciling (0/1)
        [30]  phase_filing (0/1)
        [31]  episode_progress (step / max_steps)
    """

    OBS_DIM    = 32
    N_ACTIONS  = 10   # len(ActionType)
    MAX_STEPS  = {1: 50, 2: 150, 3: 300}

    INVOICE_TYPES = ["B2B", "B2C", "EXPORT", "RCM", "ISD", "EXEMPT"]
    MISMATCH_TYPES = [
        "amount_diff", "gstin_missing", "not_in_2b",
        "rate_mismatch", "cancelled", "duplicate",
    ]
    SLABS = ["0", "5", "12", "18", "28", "exempt"]

    def __init__(self, task_id: int = 1, split: str = "train"):
        super().__init__()
        from env.gst_env import GSTEnvironment
        self._env     = GSTEnvironment(split=split)
        self._task_id = task_id
        self._max_s   = self.MAX_STEPS[task_id]
        self._last_reward = 0.0

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=10.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        self._last_reward = 0.0
        obs, info = self._env.reset(seed=seed, task_id=self._task_id)
        return self._flatten(obs), info

    def step(self, action_idx: int):
        obs_prev = self._env._build_observation()
        gst_action = self._idx_to_action(action_idx, obs_prev)
        obs, reward, terminated, truncated, info = self._env.step(gst_action)
        self._last_reward = reward
        return self._flatten(obs), reward, terminated, truncated, info

    def state(self):
        return self._env.state()

    # ──────────────────────────────────────────────────────────────────────────
    # Observation flattening
    # ──────────────────────────────────────────────────────────────────────────

    def _flatten(self, obs: GSTObservation) -> np.ndarray:
        vec = np.zeros(self.OBS_DIM, dtype=np.float32)
        ms  = self._max_s

        vec[0]  = (obs.task_id - 1) / 2.0
        vec[1]  = obs.step_number / ms
        vec[2]  = min(obs.pending_count / 50.0, 1.0)
        vec[3]  = min(obs.classified_count / 50.0, 1.0)
        vec[4]  = min(obs.flagged_count / 50.0, 1.0)
        vec[5]  = np.clip(obs.cumulative_reward / 10.0, -1.0, 1.0)
        vec[6]  = obs.steps_remaining / ms
        vec[7]  = min(obs.matched_itc_amount / 1e6, 1.0)
        vec[8]  = min(obs.disputed_itc_amount / 1e6, 1.0)

        gstr = obs.gstr3b
        vec[9]  = min(gstr.taxable_outward / 1e7, 1.0)
        vec[10] = min(gstr.itc_igst / 1e6, 1.0)
        vec[11] = min(gstr.itc_ineligible / 1e6, 1.0)
        vec[12] = min(gstr.igst_payable / 1e6, 1.0)
        vec[13] = min(gstr.cgst_payable / 1e6, 1.0)
        vec[14] = min(gstr.sgst_payable / 1e6, 1.0)
        vec[15] = min(gstr.net_payable / 1e6, 1.0)
        vec[16] = len(gstr.sections_completed) / 9.0

        inv = obs.current_invoice
        if inv:
            vec[17] = min(inv.taxable_value / 1e6, 1.0)
            vec[18] = min(inv.igst_amount / 1e5, 1.0)
            try:
                slab_str = str(inv.hsn_code or "")
                vec[19] = 0.0  # default; computed from HSN if needed
            except Exception:
                vec[19] = 0.0
            try:
                vec[20] = self.INVOICE_TYPES.index(inv.invoice_type) / len(self.INVOICE_TYPES)
            except ValueError:
                vec[20] = 0.0
            vec[21] = 1.0 if inv.flags else 0.0
            vec[22] = 1.0 if "fake_invoice" in inv.flags else 0.0
            vec[23] = 1.0 if "missing_gstin" in inv.flags else 0.0

        if obs.mismatches:
            top = obs.mismatches[0]
            vec[24] = min(abs(top.delta) / 1e5, 1.0)
            try:
                vec[25] = self.MISMATCH_TYPES.index(top.mismatch_type) / len(self.MISMATCH_TYPES)
            except ValueError:
                vec[25] = 0.0

        profile = obs.known_supplier_profile
        vec[26] = float(profile.get("historical_compliance_rate", 0.5)) if profile else 0.5

        vec[27] = 1.0 if self._last_reward > 0 else 0.0

        state = self._env.state()
        phase = state.get("phase", "CLASSIFYING")
        vec[28] = 1.0 if phase == "CLASSIFYING" else 0.0
        vec[29] = 1.0 if phase == "RECONCILING" else 0.0
        vec[30] = 1.0 if phase == "FILING"       else 0.0
        vec[31] = obs.step_number / ms

        return vec

    # ──────────────────────────────────────────────────────────────────────────
    # Action mapping: discrete index → typed GSTAction
    # ──────────────────────────────────────────────────────────────────────────

    def _idx_to_action(self, idx: int, obs: GSTObservation) -> GSTAction:
        """
        Map SB3 discrete action index to a domain-appropriate GSTAction.
        Indices align with ActionType enum order.
        """
        ts = datetime.now(timezone.utc).isoformat()
        inv = obs.current_invoice

        action_types = list(ActionType)
        if idx >= len(action_types):
            idx = 0
        atype = action_types[idx]

        try:
            if atype == ActionType.CLASSIFY_INVOICE:
                payload = ClassifyInvoicePayload(
                    invoice_id=inv.invoice_id if inv else "none",
                    invoice_type=inv.invoice_type if inv else "B2B",
                    hsn_code=inv.hsn_code or "9983" if inv else "9983",
                    gst_slab="18",
                    supply_type="goods",
                    itc_eligible=True,
                    reverse_charge=(inv.invoice_type == "RCM") if inv else False,
                )
            elif atype == ActionType.MATCH_ITC:
                if obs.mismatches:
                    top = obs.mismatches[0]
                    payload = MatchITCPayload(
                        purchase_invoice_id=top.purchase_invoice_id,
                        gstr2b_invoice_id=top.gstr2b_invoice_id or "unknown",
                        confidence=0.85,
                    )
                else:
                    atype   = ActionType.SKIP_INVOICE
                    payload = SkipInvoicePayload(invoice_id="none", reason="no mismatches")
            elif atype == ActionType.FLAG_DISCREPANCY:
                ref_id = (
                    obs.mismatches[0].purchase_invoice_id if obs.mismatches
                    else (inv.invoice_id if inv else "none")
                )
                mtype = obs.mismatches[0].mismatch_type if obs.mismatches else "amount_diff"
                payload = FlagDiscrepancyPayload(
                    invoice_id=ref_id,
                    discrepancy_type=mtype,
                    recommended_action="dispute",
                    notes="RL agent flag",
                )
            elif atype == ActionType.ACCEPT_MISMATCH:
                ref_id = obs.mismatches[0].purchase_invoice_id if obs.mismatches else "none"
                payload = AcceptMismatchPayload(purchase_invoice_id=ref_id, reason="RL accept")
            elif atype == ActionType.DEFER_INVOICE:
                ref_id = (
                    obs.mismatches[0].purchase_invoice_id if obs.mismatches
                    else (inv.invoice_id if inv else "none")
                )
                payload = DeferInvoicePayload(invoice_id=ref_id, reason="RL defer")
            elif atype == ActionType.SET_SECTION_VALUE:
                gstr  = obs.gstr3b
                done  = set(gstr.sections_completed)
                nxt   = next((s for s in GSTR3B_SECTION_ORDER if s not in done), "6.3")
                payload = SetSectionValuePayload(section=nxt, value=0.0)
            elif atype == ActionType.GENERATE_RETURN:
                payload = GenerateReturnPayload(tax_period=obs.tax_period)
            elif atype == ActionType.SUBMIT_RETURN:
                payload = SubmitReturnPayload(tax_period=obs.tax_period, declaration=True)
            elif atype == ActionType.REQUEST_CLARIFICATION:
                ref_id = inv.invoice_id if inv else "none"
                # REQUEST_CLARIFICATION is not a terminal action; treat as skip to avoid waste
                atype   = ActionType.SKIP_INVOICE
                payload = SkipInvoicePayload(invoice_id=ref_id, reason="clarification redirect")
            else:  # SKIP_INVOICE
                ref_id = inv.invoice_id if inv else "none"
                payload = SkipInvoicePayload(invoice_id=ref_id, reason="RL skip")

        except Exception:
            atype   = ActionType.SKIP_INVOICE
            payload = SkipInvoicePayload(invoice_id="error", reason="action build failed")

        return GSTAction(action_type=atype, payload=payload, timestamp=ts)


if SB3_AVAILABLE:
    # ─────────────────────────────────────────────────────────────────────────
    # Logging callback (only defined when SB3 is installed)
    # ─────────────────────────────────────────────────────────────────────────

    class EpisodeLogCallback(BaseCallback):  # type: ignore[misc]
        """Logs mean reward and episode length every N steps."""

        def __init__(self, log_interval: int = 10000, verbose: int = 0):
            super().__init__(verbose)
            self._log_interval = log_interval
            self._ep_rewards: list[float] = []
            self._ep_lengths: list[int]   = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self._ep_rewards.append(info["episode"]["r"])
                    self._ep_lengths.append(info["episode"]["l"])

            if self.n_calls % self._log_interval == 0 and self._ep_rewards:
                mean_r = sum(self._ep_rewards[-100:]) / min(len(self._ep_rewards), 100)
                mean_l = sum(self._ep_lengths[-100:]) / min(len(self._ep_lengths), 100)
                print(
                    f"[PPO] steps={self.num_timesteps:,} "
                    f"mean_reward={mean_r:.3f} mean_ep_len={mean_l:.1f}",
                    flush=True,
                )
            return True


# ─────────────────────────────────────────────────────────────────────────────
# RLAgent
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    PPO-based RL Agent.

    Trains one policy per task (three separate models).
    Each model uses a MlpPolicy with the flattened 32-dim observation.
    """

    def __init__(self, task_id: int = 1, model_path: Optional[str] = None):
        self._task_id = task_id
        self._model: Optional["PPO"] = None
        self._section_generated = False

        if model_path and SB3_AVAILABLE:
            self._model = PPO.load(model_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        task_id:    int   = 1,
        timesteps:  int   = 500_000,
        n_envs:     int   = 2,           # matches 2 vCPU constraint
        save_path:  str   = "checkpoints",
    ) -> "RLAgent":
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed. Run: pip install stable-baselines3")

        CHECKPOINTS_DIR.mkdir(exist_ok=True)

        def make_env():
            def _init():
                return FlatGSTEnv(task_id=task_id, split="train")
            return _init

        vec_env = make_vec_env(make_env(), n_envs=n_envs)

        # Eval environment (val split, deterministic)
        eval_env = FlatGSTEnv(task_id=task_id, split="val")

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=None,  # disabled for resource constraints
            verbose=0,
        )

        callbacks = [
            EpisodeLogCallback(log_interval=10_000),
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(CHECKPOINTS_DIR),
                name_prefix=f"ppo_gst_task{task_id}",
            ),
            EvalCallback(
                eval_env,
                eval_freq=25_000,
                n_eval_episodes=10,
                best_model_save_path=str(CHECKPOINTS_DIR / f"best_task{task_id}"),
                verbose=0,
            ),
        ]

        print(f"Training PPO for task {task_id}, timesteps={timesteps:,}, n_envs={n_envs}")
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

        final_path = str(CHECKPOINTS_DIR / f"ppo_gst_task{task_id}_final")
        model.save(final_path)
        print(f"Model saved to {final_path}")

        agent = cls(task_id=task_id)
        agent._model = model
        return agent

    @classmethod
    def load(cls, model_path: str, task_id: int = 1) -> "RLAgent":
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed")
        agent = cls(task_id=task_id)
        agent._model = PPO.load(model_path)
        return agent

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, obs: GSTObservation, task_id: int) -> GSTAction:
        """
        Choose action for the given observation.

        If no trained model is available, falls back to the baseline agent.
        """
        if self._model is None:
            from agent.baseline_agent import BaselineAgent
            return BaselineAgent().act(obs, task_id)

        # Build flat env just to use _flatten()
        flat_env = FlatGSTEnv(task_id=task_id)
        flat_obs = flat_env._flatten(obs)
        action_idx, _ = self._model.predict(flat_obs, deterministic=True)
        return flat_env._idx_to_action(int(action_idx), obs)

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        task_id:   int = 1,
        n_episodes: int = 10,
        split:     str  = "val",
    ) -> dict:
        """
        Evaluate the agent on `n_episodes` from the given split.
        Returns dict of mean/std reward and per-task grader scores.
        """
        from env.gst_env import GSTEnvironment
        from env.graders import ClassificationGrader, ITCGrader, FilingGrader

        env = GSTEnvironment(split=split)
        graders = {1: ClassificationGrader(), 2: ITCGrader(), 3: FilingGrader()}

        episode_rewards = []
        episode_steps   = []
        grader_scores   = []

        for ep in range(n_episodes):
            obs, _ = env.reset(task_id=task_id, episode_idx=ep)
            self._section_generated = False
            ep_reward = 0.0
            steps = 0
            done = False
            predictions, ground_truths = [], []

            while not done:
                action  = self.act(obs, task_id)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                steps += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_steps.append(steps)

            # Grader score using final state
            state = env.state()
            if task_id == 1:
                preds = [
                    {"gst_slab": state["ground_truth"].get(inv.invoice_id, {}).get("gst_slab"),
                     "hsn_code": state["ground_truth"].get(inv.invoice_id, {}).get("hsn_code"),
                     "invoice_type": inv.invoice_type,
                     "itc_eligible": state["ground_truth"].get(inv.invoice_id, {}).get("itc_eligible"),
                     "reverse_charge": state["ground_truth"].get(inv.invoice_id, {}).get("reverse_charge")}
                    for inv in state.get("invoices", [])
                ]
                gts = list(state["ground_truth"].values())
                score = graders[1].grade_batch(preds, gts) if preds and gts else 0.0
            elif task_id == 2:
                score = 0.0  # scored by environment reward
            else:
                score = graders[3].grade(
                    prediction=state.get("section_values", {}),
                    ground_truth=state.get("true_section_values", {}),
                )
            grader_scores.append(score)

        mean_r = float(np.mean(episode_rewards))
        std_r  = float(np.std(episode_rewards))

        return {
            "task_id":       task_id,
            "n_episodes":    n_episodes,
            "mean_reward":   round(mean_r, 4),
            "std_reward":    round(std_r, 4),
            "mean_steps":    round(float(np.mean(episode_steps)), 1),
            "mean_grader":   round(float(np.mean(grader_scores)), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GST RL Agent — Train / Eval")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--task",       type=int,  default=1, choices=[1, 2, 3])
    parser.add_argument("--timesteps",  type=int,  default=500_000)
    parser.add_argument("--n-envs",     type=int,  default=2)
    parser.add_argument("--model",      type=str,  default=None, help="Path to saved model")
    parser.add_argument("--n-episodes", type=int,  default=10)
    parser.add_argument("--split",      type=str,  default="val")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if args.train:
        agent = RLAgent.train(
            task_id=args.task,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
        )
        result = agent.evaluate(task_id=args.task, n_episodes=args.n_episodes)
        print(json.dumps(result, indent=2))

    elif args.eval:
        if not args.model:
            print("--model required for --eval")
            sys.exit(1)
        agent  = RLAgent.load(args.model, task_id=args.task)
        result = agent.evaluate(task_id=args.task, n_episodes=args.n_episodes, split=args.split)
        print(json.dumps(result, indent=2))
