"""
baseline_agent.py — Deterministic Rule-Based Baseline Agent

This agent uses hard-coded GST rules to make decisions without any learning.
It serves as the performance floor that the RL agent must beat.

Usage:
    from agent.baseline_agent import BaselineAgent
    from env.gst_env import GSTEnvironment

    env   = GSTEnvironment()
    agent = BaselineAgent()
    obs, _ = env.reset(task_id=1, seed=42)
    action  = agent.act(obs, task_id=1)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.observation import GSTObservation
from models.action import (
    GSTAction, ActionType,
    ClassifyInvoicePayload, MatchITCPayload,
    FlagDiscrepancyPayload, AcceptMismatchPayload,
    DeferInvoicePayload, SetSectionValuePayload,
    GenerateReturnPayload, SubmitReturnPayload,
    SkipInvoicePayload,
)

# ─────────────────────────────────────────────────────────────────────────────
# HSN lookup (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

_HSN_TABLE_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "hsn_table.json"

def _load_hsn_lookup() -> dict[str, dict]:
    try:
        with open(_HSN_TABLE_PATH, "r") as f:
            entries = json.load(f)
        return {e["hsn"]: e for e in entries}
    except FileNotFoundError:
        return {}

_HSN_LOOKUP: dict[str, dict] = _load_hsn_lookup()

GSTR3B_SECTION_ORDER = [
    "3.1a", "3.1b", "3.1c", "3.1d",
    "4a", "4b",
    "6.1", "6.2", "6.3",
]


# ─────────────────────────────────────────────────────────────────────────────
# GSTIN state code helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_inter_state(gstin_a: Optional[str], gstin_b: Optional[str]) -> bool:
    if not gstin_a or not gstin_b or len(gstin_a) < 2 or len(gstin_b) < 2:
        return True  # assume IGST when unknown
    return gstin_a[:2] != gstin_b[:2]


# ─────────────────────────────────────────────────────────────────────────────
# BaselineAgent
# ─────────────────────────────────────────────────────────────────────────────

class BaselineAgent:
    """
    Deterministic rule-based agent.

    Task 1 — Invoice Classification:
        Uses HSN lookup table to determine slab and supply type.
        Detects missing GSTIN → B2C + no ITC.
        Detects RCM from invoice_type field.

    Task 2 — ITC Reconciliation:
        not_in_2b                → defer_invoice
        delta < 1,000            → accept_mismatch
        delta > 10,000           → flag_discrepancy (dispute)
        fake_invoice flag        → flag_discrepancy (dispute)
        cancelled / duplicate    → flag_discrepancy (hold_itc / write_off)
        otherwise                → match_itc if GSTIN matches

    Task 3 — GSTR-3B Filing:
        Fills sections in legal order using matched ITC from observation.
        Applies ITC offset rules: IGST → IGST/CGST/SGST, CGST → CGST, SGST → SGST.
    """

    def __init__(self):
        self._section_idx: int = 0    # tracks filing progress within an episode
        self._generated: bool  = False

    def reset(self):
        """Call at episode start for Task 3 to reset section pointer."""
        self._section_idx = 0
        self._generated   = False

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, obs: GSTObservation, task_id: int) -> GSTAction:
        ts = datetime.now(timezone.utc).isoformat()
        if task_id == 1:
            return self._act_classify(obs, ts)
        elif task_id == 2:
            return self._act_itc(obs, ts)
        else:
            return self._act_filing(obs, ts)

    # ──────────────────────────────────────────────────────────────────────────
    # Task 1 — Invoice Classification
    # ──────────────────────────────────────────────────────────────────────────

    def _act_classify(self, obs: GSTObservation, ts: str) -> GSTAction:
        inv = obs.current_invoice
        if inv is None:
            return GSTAction(
                action_type=ActionType.SKIP_INVOICE,
                payload=SkipInvoicePayload(invoice_id="none", reason="no current invoice"),
                timestamp=ts,
            )

        # Detect fake invoice (invalid GSTIN flags)
        if "fake_invoice" in inv.flags:
            return GSTAction(
                action_type=ActionType.FLAG_DISCREPANCY,
                payload=FlagDiscrepancyPayload(
                    invoice_id=inv.invoice_id,
                    discrepancy_type="fake_invoice",
                    recommended_action="dispute",
                    notes="Invalid GSTIN checksum — suspected fake invoice",
                ),
                timestamp=ts,
                agent_reasoning="fake_invoice flag detected",
            )

        # Missing supplier GSTIN → B2C
        if not inv.supplier_gstin or "missing_gstin" in inv.flags:
            payload = ClassifyInvoicePayload(
                invoice_id=inv.invoice_id,
                invoice_type="B2C",
                hsn_code=inv.hsn_code or "9999",
                gst_slab=self._hsn_to_slab(inv.hsn_code),
                supply_type=self._hsn_to_supply(inv.hsn_code),
                itc_eligible=False,
                reverse_charge=False,
            )
            return GSTAction(
                action_type=ActionType.CLASSIFY_INVOICE, payload=payload, timestamp=ts,
                agent_reasoning="missing supplier GSTIN → B2C, no ITC",
            )

        # Determine invoice type from observation field (trust supplier declaration)
        inv_type = inv.invoice_type or "B2B"
        is_rcm   = (inv_type == "RCM")
        is_export = (inv_type == "EXPORT")

        # ITC eligibility
        # RCM: ITC only after tax paid
        # EXPORT: zero-rated, ITC refund eligible
        # B2B: eligible if not Section 17(5)
        itc_eligible = (
            inv_type in ("B2B", "RCM", "ISD", "EXPORT")
            and "fake_invoice" not in inv.flags
        )

        # GST slab from HSN table
        gst_slab    = self._hsn_to_slab(inv.hsn_code)
        supply_type = self._hsn_to_supply(inv.hsn_code)

        payload = ClassifyInvoicePayload(
            invoice_id=inv.invoice_id,
            invoice_type=inv_type,
            hsn_code=inv.hsn_code or "9999",
            gst_slab="0" if is_export else gst_slab,
            supply_type=supply_type,
            itc_eligible=itc_eligible,
            reverse_charge=is_rcm,
        )
        return GSTAction(
            action_type=ActionType.CLASSIFY_INVOICE, payload=payload, timestamp=ts,
            agent_reasoning=f"HSN={inv.hsn_code} slab={gst_slab} type={inv_type}",
        )

    def _hsn_to_slab(self, hsn: Optional[str]) -> str:
        if not hsn:
            return "18"
        entry = _HSN_LOOKUP.get(hsn) or _HSN_LOOKUP.get(hsn[:4])
        return str(entry["slab"]) if entry else "18"

    def _hsn_to_supply(self, hsn: Optional[str]) -> str:
        if not hsn:
            return "services"
        entry = _HSN_LOOKUP.get(hsn) or _HSN_LOOKUP.get(hsn[:4])
        return entry.get("supply_type", "goods") if entry else "goods"

    # ──────────────────────────────────────────────────────────────────────────
    # Task 2 — ITC Reconciliation
    # ──────────────────────────────────────────────────────────────────────────

    def _act_itc(self, obs: GSTObservation, ts: str) -> GSTAction:
        if not obs.mismatches:
            return GSTAction(
                action_type=ActionType.SKIP_INVOICE,
                payload=SkipInvoicePayload(invoice_id="done", reason="no mismatches remaining"),
                timestamp=ts,
            )

        top     = obs.mismatches[0]
        inv_id  = top.purchase_invoice_id
        g_id    = top.gstr2b_invoice_id
        mtype   = top.mismatch_type
        delta   = top.delta

        # not_in_2b → defer and wait for supplier
        if mtype == "not_in_2b" or g_id is None:
            return GSTAction(
                action_type=ActionType.DEFER_INVOICE,
                payload=DeferInvoicePayload(
                    invoice_id=inv_id,
                    reason="Invoice not in GSTR-2B — awaiting supplier filing",
                ),
                timestamp=ts,
                agent_reasoning="not_in_2b → defer",
            )

        # fake_invoice → flag and dispute
        if mtype == "fake_invoice":
            return GSTAction(
                action_type=ActionType.FLAG_DISCREPANCY,
                payload=FlagDiscrepancyPayload(
                    invoice_id=inv_id,
                    discrepancy_type="fake_invoice",
                    recommended_action="dispute",
                    notes="Fake invoice detected — dispute with supplier",
                ),
                timestamp=ts,
                agent_reasoning="fake_invoice → dispute",
            )

        # cancelled → hold ITC
        if mtype == "cancelled":
            return GSTAction(
                action_type=ActionType.FLAG_DISCREPANCY,
                payload=FlagDiscrepancyPayload(
                    invoice_id=inv_id,
                    discrepancy_type="cancelled",
                    recommended_action="hold_itc",
                    notes="Invoice cancelled — hold ITC until resolved",
                ),
                timestamp=ts,
                agent_reasoning="cancelled → hold_itc",
            )

        # duplicate → write off
        if mtype == "duplicate":
            return GSTAction(
                action_type=ActionType.FLAG_DISCREPANCY,
                payload=FlagDiscrepancyPayload(
                    invoice_id=inv_id,
                    discrepancy_type="duplicate",
                    recommended_action="write_off",
                    notes="Duplicate invoice in purchase register",
                ),
                timestamp=ts,
                agent_reasoning="duplicate → write_off",
            )

        # amount_diff or rate_mismatch
        if mtype in ("amount_diff", "rate_mismatch"):
            if delta < 1000:
                return GSTAction(
                    action_type=ActionType.ACCEPT_MISMATCH,
                    payload=AcceptMismatchPayload(
                        purchase_invoice_id=inv_id,
                        reason=f"Immaterial difference: delta={delta:.2f} < 1000",
                    ),
                    timestamp=ts,
                    agent_reasoning="delta < 1000 → accept",
                )
            elif delta > 10000:
                return GSTAction(
                    action_type=ActionType.FLAG_DISCREPANCY,
                    payload=FlagDiscrepancyPayload(
                        invoice_id=inv_id,
                        discrepancy_type=mtype,
                        recommended_action="dispute",
                        notes=f"Material difference: delta={delta:.2f} — raise dispute",
                    ),
                    timestamp=ts,
                    agent_reasoning="delta > 10000 → dispute",
                )
            else:
                # 1000–10000: defer for clarification
                return GSTAction(
                    action_type=ActionType.DEFER_INVOICE,
                    payload=DeferInvoicePayload(
                        invoice_id=inv_id,
                        reason=f"Moderate difference delta={delta:.2f} — defer pending clarification",
                    ),
                    timestamp=ts,
                    agent_reasoning="1000 < delta < 10000 → defer",
                )

        # Default: attempt match if GSTIN matches
        if g_id:
            return GSTAction(
                action_type=ActionType.MATCH_ITC,
                payload=MatchITCPayload(
                    purchase_invoice_id=inv_id,
                    gstr2b_invoice_id=g_id,
                    confidence=0.80,
                ),
                timestamp=ts,
                agent_reasoning="default match",
            )

        return GSTAction(
            action_type=ActionType.SKIP_INVOICE,
            payload=SkipInvoicePayload(invoice_id=inv_id, reason="unhandled mismatch type"),
            timestamp=ts,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 3 — GSTR-3B Filing
    # ──────────────────────────────────────────────────────────────────────────

    def _act_filing(self, obs: GSTObservation, ts: str) -> GSTAction:
        gstr            = obs.gstr3b
        sections_done   = set(gstr.sections_completed)
        matched_itc     = obs.matched_itc_amount
        tax_period      = obs.tax_period

        # Fill next section in legal order
        for sec in GSTR3B_SECTION_ORDER:
            if sec in sections_done:
                continue
            value = self._estimate_section(sec, obs)
            return GSTAction(
                action_type=ActionType.SET_SECTION_VALUE,
                payload=SetSectionValuePayload(section=sec, value=value),
                timestamp=ts,
                agent_reasoning=f"filling {sec}={value:.2f}",
            )

        # All sections filled — generate return
        if not self._generated:
            self._generated = True
            return GSTAction(
                action_type=ActionType.GENERATE_RETURN,
                payload=GenerateReturnPayload(tax_period=tax_period),
                timestamp=ts,
                agent_reasoning="all sections filled, generating return",
            )

        # Submit
        return GSTAction(
            action_type=ActionType.SUBMIT_RETURN,
            payload=SubmitReturnPayload(tax_period=tax_period, declaration=True),
            timestamp=ts,
            agent_reasoning="return generated, submitting",
        )

    def _estimate_section(self, section: str, obs: GSTObservation) -> float:
        """
        Estimate section value from observable state.
        Uses matched ITC and applies legal ITC offset rules.
        """
        gstr        = obs.gstr3b
        matched_itc = obs.matched_itc_amount

        section_map = {
            "3.1a": gstr.taxable_outward,
            "3.1b": gstr.zero_rated,
            "3.1c": gstr.exempted,
            "3.1d": gstr.rcm_inward,
            "4a":   matched_itc,
            "4b":   gstr.itc_ineligible,
            "6.1":  max(0.0, gstr.igst_payable),
            "6.2":  max(0.0, gstr.cgst_payable),
            "6.3":  max(0.0, gstr.sgst_payable),
        }
        return round(section_map.get(section, 0.0), 2)
