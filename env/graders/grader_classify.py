from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, prediction: dict, ground_truth: dict) -> float:
        """Returns deterministic score ∈ [0.0, 1.0]"""
        ...

    def is_deterministic(self) -> bool:
        return True


@dataclass
class ClassificationPrediction:
    invoice_id:     str
    invoice_type:   str
    hsn_code:       str
    gst_slab:       str
    supply_type:    str
    itc_eligible:   bool
    reverse_charge: bool


class ClassificationGrader(BaseGrader):
    """
    Deterministic grader for Task 1 — Invoice Classification.

    Weights:
        gst_slab       0.40  (wrong slab → no partial credit)
        hsn_code       0.25  (4-digit prefix match → 0.125)
        invoice_type   0.15
        itc_eligible   0.10
        reverse_charge 0.10
    """

    WEIGHTS = {
        "gst_slab":       0.40,
        "hsn_code":       0.25,
        "invoice_type":   0.15,
        "itc_eligible":   0.10,
        "reverse_charge": 0.10,
    }

    def grade(self, prediction: dict, ground_truth: dict) -> float:
        score = 0.0

        # GST slab — exact match only
        if prediction.get("gst_slab") == ground_truth.get("gst_slab"):
            score += self.WEIGHTS["gst_slab"]

        # HSN code — full or 4-digit prefix
        pred_hsn = str(prediction.get("hsn_code", ""))
        true_hsn = str(ground_truth.get("hsn_code", ""))
        if pred_hsn == true_hsn:
            score += self.WEIGHTS["hsn_code"]
        elif len(pred_hsn) >= 4 and len(true_hsn) >= 4 and pred_hsn[:4] == true_hsn[:4]:
            score += self.WEIGHTS["hsn_code"] * 0.5

        # Invoice type
        if prediction.get("invoice_type") == ground_truth.get("invoice_type"):
            score += self.WEIGHTS["invoice_type"]

        # ITC eligible
        if bool(prediction.get("itc_eligible")) == bool(ground_truth.get("itc_eligible")):
            score += self.WEIGHTS["itc_eligible"]

        # Reverse charge
        if bool(prediction.get("reverse_charge")) == bool(ground_truth.get("reverse_charge")):
            score += self.WEIGHTS["reverse_charge"]

        return round(score, 4)

    def grade_batch(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        """Average score over a batch of invoice classifications."""
        if not predictions or not ground_truths:
            return 0.0
        n = min(len(predictions), len(ground_truths))
        scores = [self.grade(predictions[i], ground_truths[i]) for i in range(n)]
        return round(sum(scores) / n, 4)
