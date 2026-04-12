from __future__ import annotations

from collections import defaultdict
from typing import Optional


class MemoryManager:
    """
    Per-episode memory: tracks past invoice decisions and supplier profiles
    so the agent can retrieve similar_past_invoices and known_supplier_profile
    fields in each observation.
    """

    def __init__(self):
        self._invoice_history: list[dict] = []
        self._supplier_profiles: dict[str, dict] = {}
        self._action_counts: dict[str, int] = defaultdict(int)   # for loop detection

    # ──────────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        self._invoice_history.clear()
        self._supplier_profiles.clear()
        self._action_counts.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Invoice history
    # ──────────────────────────────────────────────────────────────────────────

    def record_decision(
        self,
        invoice_id: str,
        hsn: Optional[str],
        decision: str,
        outcome: str,
        supplier_gstin: Optional[str] = None,
    ):
        """Store a resolved invoice decision for future retrieval."""
        self._invoice_history.append({
            "invoice_id": invoice_id,
            "hsn": hsn,
            "decision": decision,
            "outcome": outcome,
            "supplier_gstin": supplier_gstin,
        })
        # Track action counts for duplicate-action detection
        key = f"{invoice_id}:{decision}"
        self._action_counts[key] += 1

    def is_duplicate_action(self, invoice_id: str, decision: str) -> bool:
        key = f"{invoice_id}:{decision}"
        return self._action_counts[key] > 0

    def get_similar_invoices(self, hsn: Optional[str], limit: int = 3) -> list[dict]:
        """Return past invoices with the same 4-digit HSN prefix."""
        if not hsn:
            return self._invoice_history[-limit:]
        prefix = hsn[:4]
        matches = [
            rec for rec in self._invoice_history
            if rec.get("hsn") and rec["hsn"][:4] == prefix
        ]
        return matches[-limit:] if matches else self._invoice_history[-limit:]

    # ──────────────────────────────────────────────────────────────────────────
    # Supplier profiles
    # ──────────────────────────────────────────────────────────────────────────

    def update_supplier_profile(self, gstin: str, compliance_rate: float,
                                avg_delay_days: float, cancelled_pct: float):
        self._supplier_profiles[gstin] = {
            "gstin": gstin,
            "historical_compliance_rate": round(compliance_rate, 4),
            "avg_delay_days": round(avg_delay_days, 2),
            "cancelled_invoices_pct": round(cancelled_pct, 4),
        }

    def get_supplier_profile(self, gstin: Optional[str]) -> Optional[dict]:
        if not gstin:
            return None
        return self._supplier_profiles.get(gstin)
