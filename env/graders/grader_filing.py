from __future__ import annotations

from env.graders.grader_classify import BaseGrader

GSTR3B_SECTIONS = ["3.1a", "3.1b", "3.1c", "3.1d", "4a", "4b", "6.1", "6.2", "6.3"]


class FilingGrader(BaseGrader):
    """
    Deterministic grader for Task 3 — GSTR-3B Filing.

    Per-section scoring:
        pct_error < 0.05   → +0.20
        pct_error < 0.20   → +0.10
        otherwise          → −0.10 (not added, base = 0)

    Terminal penalties:
        net underpayment > 2%   → −0.30
        ITC overclaim > 2%      → −0.50
    """

    def grade(self, prediction: dict, ground_truth: dict) -> float:
        """
        prediction keys:  section values dict  {section → float}
        ground_truth keys: same structure + total_itc, net_payable
        """
        section_scores: dict[str, float] = {}

        for sec in GSTR3B_SECTIONS:
            agent_val = prediction.get(sec, 0.0)
            true_val  = ground_truth.get(sec, 0.0)
            pct_err   = abs(agent_val - true_val) / (abs(true_val) + 1.0)

            if pct_err < 0.05:
                section_scores[sec] = 0.20
            elif pct_err < 0.20:
                section_scores[sec] = 0.10
            else:
                section_scores[sec] = 0.0

        base = sum(section_scores.values()) / max(len(GSTR3B_SECTIONS), 1)

        # Net payable underpayment penalty
        agent_net = prediction.get("net_payable", 0.0)
        true_net  = ground_truth.get("net_payable", 0.0)
        underpay = -0.30 if true_net > 0 and agent_net < true_net * 0.98 else 0.0

        # ITC overclaim penalty
        agent_itc = prediction.get("total_itc", 0.0)
        true_itc  = ground_truth.get("total_itc", 0.0)
        overclaim = -0.50 if true_itc > 0 and agent_itc > true_itc * 1.02 else 0.0

        final = base + underpay + overclaim
        return round(max(0.0, min(1.0, final)), 4)

    def section_breakdown(self, prediction: dict, ground_truth: dict) -> dict[str, float]:
        """Per-section accuracy breakdown for audit/debug."""
        breakdown = {}
        for sec in GSTR3B_SECTIONS:
            agent_val = prediction.get(sec, 0.0)
            true_val  = ground_truth.get(sec, 0.0)
            pct_err   = abs(agent_val - true_val) / (abs(true_val) + 1.0)
            breakdown[sec] = round(1.0 - pct_err, 4)
        return breakdown
