from __future__ import annotations

from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz

from env.graders.grader_classify import BaseGrader


def itc_match_score(purchase: dict, gstr2b: dict) -> float:
    """
    Composite similarity score ∈ [0, 1] between a purchase invoice and a GSTR-2B entry.

    Components:
        GSTIN match         0.35
        Amount similarity   0.35  (within 10% delta)
        Date proximity      0.15  (within 30 days)
        Invoice number      0.15  (fuzzy ratio)
    """
    # GSTIN
    gstin_score = 1.0 if purchase.get("supplier_gstin") == gstr2b.get("supplier_gstin") else 0.0

    # Amount delta
    p_val = purchase.get("taxable_value", 0.0)
    g_val = gstr2b.get("taxable_value", 0.0)
    delta_pct = abs(p_val - g_val) / (g_val + 1.0)
    amount_score = max(0.0, 1.0 - delta_pct / 0.10)

    # Date proximity
    date_score = 0.0
    try:
        p_date = datetime.strptime(purchase.get("invoice_date", ""), "%Y-%m-%d")
        g_date = datetime.strptime(gstr2b.get("invoice_date", ""), "%Y-%m-%d")
        date_diff = abs((p_date - g_date).days)
        date_score = max(0.0, 1.0 - date_diff / 30.0)
    except (ValueError, TypeError):
        date_score = 0.0

    # Invoice number fuzzy match
    num_sim = fuzz.ratio(
        str(purchase.get("invoice_number", "")),
        str(gstr2b.get("invoice_number", "")),
    ) / 100.0

    return round(0.35 * gstin_score + 0.35 * amount_score + 0.15 * date_score + 0.15 * num_sim, 4)


class ITCGrader(BaseGrader):
    """
    Deterministic grader for Task 2 — ITC Reconciliation.

    Grades the full reconciliation result: matched ITC, flagged discrepancies,
    and correctness of discrepancy type + recommended action.
    """

    def grade(self, prediction: dict, ground_truth: dict) -> float:
        """
        prediction keys:
            matched_pairs:    list of {purchase_id, gstr2b_id}
            flagged:          list of {invoice_id, discrepancy_type, recommended_action}

        ground_truth keys:
            correct_matches:  dict {purchase_id → gstr2b_id}
            correct_flags:    dict {invoice_id → {discrepancy_type, recommended_action}}
        """
        correct_matches = ground_truth.get("correct_matches", {})
        correct_flags   = ground_truth.get("correct_flags", {})
        total_items     = len(correct_matches) + len(correct_flags)
        if total_items == 0:
            return 1.0

        score = 0.0

        # Evaluate matches (50% weight of each item)
        for pair in prediction.get("matched_pairs", []):
            pid  = pair.get("purchase_id", "")
            gid  = pair.get("gstr2b_id", "")
            if correct_matches.get(pid) == gid:
                score += 0.50

        # Evaluate flags
        for flag in prediction.get("flagged", []):
            inv_id = flag.get("invoice_id", "")
            gt_flag = correct_flags.get(inv_id)
            if gt_flag is None:
                continue
            flag_score = 0.0
            if flag.get("discrepancy_type") == gt_flag.get("discrepancy_type"):
                flag_score += 0.15
            if flag.get("recommended_action") == gt_flag.get("recommended_action"):
                flag_score += 0.15
            score += flag_score

        final_score = min(0.99, score / total_items) if total_items > 0 else 0.5
        return round(max(0.01, final_score), 4)

    def grade_match_only(self, prediction: dict, ground_truth: dict) -> tuple[float, float]:
        """Returns (precision, recall) for match decisions."""
        correct_matches = ground_truth.get("correct_matches", {})
        matched_pairs   = prediction.get("matched_pairs", [])

        tp = sum(
            1 for p in matched_pairs
            if correct_matches.get(p.get("purchase_id")) == p.get("gstr2b_id")
        )
        precision = tp / max(len(matched_pairs), 1)
        recall    = tp / max(len(correct_matches), 1)
        return round(precision, 4), round(recall, 4)
