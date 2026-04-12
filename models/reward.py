from __future__ import annotations

from pydantic import BaseModel, Field


class RewardSignal(BaseModel):
    step_reward:       float                        # Immediate reward ∈ [-1, 1]
    cumulative_reward: float                        # Episode total so far
    sub_rewards:       dict[str, float] = Field(default_factory=dict)   # Component breakdown
    penalty_flags:     list[str] = Field(default_factory=list)          # Why penalties applied
    shaping_bonus:     float = 0.0                 # Potential-based shaping term
