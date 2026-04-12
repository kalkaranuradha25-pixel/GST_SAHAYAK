from __future__ import annotations

from typing import Optional

from data.synthetic.generator import SyntheticDataGenerator
from models.action import (
    GSTAction, ActionType,
    MatchITCPayload, FlagDiscrepancyPayload,
    AcceptMismatchPayload, DeferInvoicePayload,
)
from models.observation import ITCMismatch, InvoiceStatus
from env.memory_manager import MemoryManager


class Task2ITC:
    """
    Task 2 — ITC Reconciliation (Medium, max 150 steps)

    The agent matches purchase register invoices against GSTR-2B entries
    and flags discrepancies with the correct type and recommended action.
    """

    def __init__(self, seed: Optional[int] = None):
        self._gen = SyntheticDataGenerator(seed=seed)

    def get_batch(self) -> dict:
        return self._gen.generate_task2_batch()

    def apply_action(self, action: GSTAction, state: dict, memory: MemoryManager):
        t = action.action_type
        p = action.payload

        if t == ActionType.MATCH_ITC and isinstance(p, MatchITCPayload):
            purchase_map = state.get("purchase_map", {})
            gstr2b_map   = state.get("gstr2b_map", {})
            p_inv = purchase_map.get(p.purchase_invoice_id)
            g_inv = gstr2b_map.get(p.gstr2b_invoice_id)

            if p_inv and g_inv:
                taxable = p_inv.get("taxable_value", 0.0)
                igst    = p_inv.get("igst_amount", 0.0)
                state["matched_itc"] = state.get("matched_itc", 0.0) + igst
                state["total_itc_claimed"] = state.get("total_itc_claimed", 0.0) + igst
                # Remove from pending mismatches
                state["mismatches"] = [
                    m for m in state.get("mismatches", [])
                    if m.purchase_invoice_id != p.purchase_invoice_id
                ]
                memory.record_decision(
                    invoice_id=p.purchase_invoice_id,
                    hsn=p_inv.get("hsn_code"),
                    decision="matched",
                    outcome="accepted",
                    supplier_gstin=p_inv.get("supplier_gstin"),
                )
                state["pending_count"] = max(0, state.get("pending_count", 0) - 1)
                state["classified_count"] = state.get("classified_count", 0) + 1

        elif t == ActionType.FLAG_DISCREPANCY and isinstance(p, FlagDiscrepancyPayload):
            purchase_map = state.get("purchase_map", {})
            p_inv = purchase_map.get(p.invoice_id, {})
            igst  = p_inv.get("igst_amount", 0.0)
            state["disputed_itc"] = state.get("disputed_itc", 0.0) + igst
            state["flagged_count"] = state.get("flagged_count", 0) + 1
            state["mismatches"] = [
                m for m in state.get("mismatches", [])
                if m.purchase_invoice_id != p.invoice_id
            ]
            state["pending_count"] = max(0, state.get("pending_count", 0) - 1)
            memory.record_decision(
                invoice_id=p.invoice_id,
                hsn=None,
                decision=f"flagged:{p.discrepancy_type}:{p.recommended_action}",
                outcome="flagged",
            )

        elif t == ActionType.ACCEPT_MISMATCH and isinstance(p, AcceptMismatchPayload):
            purchase_map = state.get("purchase_map", {})
            p_inv = purchase_map.get(p.purchase_invoice_id, {})
            igst  = p_inv.get("igst_amount", 0.0)
            state["matched_itc"] = state.get("matched_itc", 0.0) + igst
            state["total_itc_claimed"] = state.get("total_itc_claimed", 0.0) + igst
            state["mismatches"] = [
                m for m in state.get("mismatches", [])
                if m.purchase_invoice_id != p.purchase_invoice_id
            ]
            state["pending_count"] = max(0, state.get("pending_count", 0) - 1)
            state["classified_count"] = state.get("classified_count", 0) + 1

        elif t == ActionType.DEFER_INVOICE and isinstance(p, DeferInvoicePayload):
            state["pending_count"] = max(0, state.get("pending_count", 0) - 1)
            # Deferred: no ITC claimed yet; stays in disputed
            state["mismatches"] = [
                m for m in state.get("mismatches", [])
                if m.purchase_invoice_id != p.invoice_id
            ]

        elif t == ActionType.SKIP_INVOICE:
            inv_id = getattr(p, "invoice_id", None)
            if inv_id:
                state["mismatches"] = [
                    m for m in state.get("mismatches", [])
                    if m.purchase_invoice_id != inv_id
                ]
            state["pending_count"] = max(0, state.get("pending_count", 0) - 1)

    def is_done(self, state: dict) -> bool:
        return len(state.get("mismatches", [])) == 0 or state.get("pending_count", 0) == 0
