from __future__ import annotations

from typing import Optional

from data.synthetic.generator import SyntheticDataGenerator
from models.action import GSTAction, ActionType, ClassifyInvoicePayload
from models.observation import InvoiceStatus
from env.memory_manager import MemoryManager


class Task1Classify:
    """
    Task 1 — Invoice Classification (Easy, max 50 steps)

    The agent must classify each invoice with:
        invoice_type, hsn_code, gst_slab, supply_type, itc_eligible, reverse_charge
    """

    def __init__(self, seed: Optional[int] = None):
        self._gen = SyntheticDataGenerator(seed=seed)

    def get_batch(self) -> dict:
        """Generate a batch of invoices with ground truth."""
        return self._gen.generate_task1_batch()

    def apply_action(self, action: GSTAction, state: dict, memory: MemoryManager):
        """Mutate state based on the agent's classification action."""
        t = action.action_type
        p = action.payload

        if t == ActionType.CLASSIFY_INVOICE and isinstance(p, ClassifyInvoicePayload):
            invoices = state.get("invoices", [])
            idx      = state.get("current_invoice_idx", 0)

            if idx < len(invoices):
                inv = invoices[idx]
                # Record decision in memory
                memory.record_decision(
                    invoice_id=inv.invoice_id,
                    hsn=p.hsn_code,
                    decision=f"{p.invoice_type}/{p.hsn_code}/{p.gst_slab}",
                    outcome="classified",
                    supplier_gstin=inv.supplier_gstin,
                )
                # Update invoice status
                inv.status = InvoiceStatus.CLASSIFIED
                state["classified_count"] = state.get("classified_count", 0) + 1
                state["pending_count"]    = max(0, state.get("pending_count", 0) - 1)
                state["current_invoice_idx"] = idx + 1

        elif t == ActionType.SKIP_INVOICE:
            idx = state.get("current_invoice_idx", 0)
            state["current_invoice_idx"] = idx + 1
            state["pending_count"] = max(0, state.get("pending_count", 0) - 1)

        elif t == ActionType.FLAG_DISCREPANCY:
            idx = state.get("current_invoice_idx", 0)
            invoices = state.get("invoices", [])
            if idx < len(invoices):
                invoices[idx].status = InvoiceStatus.FLAGGED
                state["flagged_count"] = state.get("flagged_count", 0) + 1
                state["pending_count"] = max(0, state.get("pending_count", 0) - 1)
                state["current_invoice_idx"] = idx + 1

    def is_done(self, state: dict) -> bool:
        """Episode ends when all invoices are processed."""
        return state.get("pending_count", 0) == 0
