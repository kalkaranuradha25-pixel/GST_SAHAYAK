from __future__ import annotations

from typing import Optional

from data.synthetic.generator import SyntheticDataGenerator
from models.action import (
    GSTAction, ActionType,
    SetSectionValuePayload, SubmitReturnPayload,
)
from env.memory_manager import MemoryManager

GSTR3B_SECTIONS = ["3.1a", "3.1b", "3.1c", "3.1d", "4a", "4b", "6.1", "6.2", "6.3"]

# Legally mandated ITC offset order
ITC_OFFSET_ORDER = {
    "igst_itc":  ["igst_payable", "cgst_payable", "sgst_payable"],
    "cgst_itc":  ["cgst_payable"],
    "sgst_itc":  ["sgst_payable"],
}


class Task3Filing:
    """
    Task 3 — GSTR-3B Filing (Hard, max 300 steps)

    The agent must fill all GSTR-3B sections in order and submit the return.
    """

    def __init__(self, seed: Optional[int] = None):
        self._gen = SyntheticDataGenerator(seed=seed)

    def get_batch(self) -> dict:
        return self._gen.generate_task3_batch()

    def apply_action(self, action: GSTAction, state: dict, memory: MemoryManager):
        t = action.action_type
        p = action.payload

        if t == ActionType.SET_SECTION_VALUE and isinstance(p, SetSectionValuePayload):
            section_values = state.setdefault("section_values", {})
            section_values[p.section] = p.value

            # Update GSTR3B summary mirror
            gstr3b = state.setdefault("gstr3b_summary", {})
            field_map = {
                "3.1a": "taxable_outward",
                "3.1b": "zero_rated",
                "3.1c": "exempted",
                "3.1d": "rcm_inward",
                "4a":   "itc_igst",
                "4b":   "itc_ineligible",
                "6.1":  "igst_payable",
                "6.2":  "cgst_payable",
                "6.3":  "sgst_payable",
            }
            if p.section in field_map:
                gstr3b[field_map[p.section]] = p.value

            sections_completed = state.setdefault("sections_completed", [])
            if p.section not in sections_completed:
                sections_completed.append(p.section)
            gstr3b["sections_completed"] = sections_completed

        elif t == ActionType.GENERATE_RETURN:
            # Compute net payable based on ITC offset rules
            sv = state.get("section_values", {})
            true_vals = state.get("true_section_values", {})

            igst_itc = sv.get("4a", 0.0)
            igst_pay = sv.get("6.1", 0.0)
            cgst_pay = sv.get("6.2", 0.0)
            sgst_pay = sv.get("6.3", 0.0)

            # Apply ITC offset: IGST ITC → IGST first, then CGST, then SGST
            igst_remaining = max(0.0, igst_pay - igst_itc)
            overflow = max(0.0, igst_itc - igst_pay)
            cgst_net = max(0.0, cgst_pay - overflow)
            overflow2 = max(0.0, overflow - cgst_pay)
            sgst_net = max(0.0, sgst_pay - overflow2)

            net_payable = igst_remaining + cgst_net + sgst_net
            state["section_values"]["net_payable"] = round(net_payable, 2)
            state.setdefault("gstr3b_summary", {})["net_payable"] = round(net_payable, 2)

            # Verify ITC offset was applied correctly
            true_net = true_vals.get("net_payable", 0.0)
            state["itc_offset_correct"] = abs(net_payable - true_net) / (abs(true_net) + 1.0) < 0.05

        elif t == ActionType.SUBMIT_RETURN and isinstance(p, SubmitReturnPayload):
            # Mark return as submitted in state; actual reward computed in RewardEngine
            state["return_submitted"] = True
            state["return_tax_period"] = p.tax_period
            state["declaration_accepted"] = p.declaration

            # Count unresolved discrepancies
            state["unresolved_discrepancies"] = len([
                m for m in state.get("mismatches", [])
            ])

    def is_done(self, state: dict) -> bool:
        """Episode ends on submit or ITC overclaim."""
        if state.get("return_submitted", False):
            return True
        # ITC overclaim is episode-ending
        true_itc  = state.get("eligible_itc", 0.0)
        agent_itc = state.get("total_itc_claimed", 0.0)
        if true_itc > 0 and agent_itc > true_itc * 1.02:
            return True
        return False
