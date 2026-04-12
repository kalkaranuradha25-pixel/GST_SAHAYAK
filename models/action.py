from __future__ import annotations

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel


class ActionType(str, Enum):
    CLASSIFY_INVOICE      = "classify_invoice"
    MATCH_ITC             = "match_itc"
    FLAG_DISCREPANCY      = "flag_discrepancy"
    ACCEPT_MISMATCH       = "accept_mismatch"
    DEFER_INVOICE         = "defer_invoice"
    SET_SECTION_VALUE     = "set_section_value"
    GENERATE_RETURN       = "generate_return"
    SUBMIT_RETURN         = "submit_return"
    REQUEST_CLARIFICATION = "request_clarification"
    SKIP_INVOICE          = "skip_invoice"


class ClassifyInvoicePayload(BaseModel):
    invoice_id:     str
    invoice_type:   str   # B2B / B2C / EXPORT / RCM / ISD / EXEMPT
    hsn_code:       str
    gst_slab:       str   # "0" / "5" / "12" / "18" / "28" / "exempt"
    supply_type:    str   # "goods" / "services"
    itc_eligible:   bool
    reverse_charge: bool


class MatchITCPayload(BaseModel):
    purchase_invoice_id: str
    gstr2b_invoice_id:   str
    confidence:          float


class FlagDiscrepancyPayload(BaseModel):
    invoice_id:         str
    discrepancy_type:   str   # amount_diff / gstin_missing / not_in_2b / rate_mismatch / cancelled / duplicate / fake_invoice
    recommended_action: str   # hold_itc / defer / dispute / write_off
    notes:              str


class AcceptMismatchPayload(BaseModel):
    purchase_invoice_id: str
    reason:              str


class DeferInvoicePayload(BaseModel):
    invoice_id: str
    reason:     str


class SetSectionValuePayload(BaseModel):
    section: str   # "3.1a" / "3.1b" / ... / "6.3"
    value:   float


class GenerateReturnPayload(BaseModel):
    tax_period: str


class SubmitReturnPayload(BaseModel):
    tax_period:  str
    declaration: bool = True


class SkipInvoicePayload(BaseModel):
    invoice_id: str
    reason:     Optional[str] = None


class RequestClarificationPayload(BaseModel):
    invoice_id: str
    question:   str


# Union of all payload types
AnyPayload = Union[
    ClassifyInvoicePayload,
    MatchITCPayload,
    FlagDiscrepancyPayload,
    AcceptMismatchPayload,
    DeferInvoicePayload,
    SetSectionValuePayload,
    GenerateReturnPayload,
    SubmitReturnPayload,
    SkipInvoicePayload,
    RequestClarificationPayload,
    dict,
]


class GSTAction(BaseModel):
    action_type:     ActionType
    payload:         AnyPayload
    timestamp:       str
    agent_reasoning: Optional[str] = None

    def to_action_str(self) -> str:
        """Compact single-line string for [STEP] action= field."""
        p = self.payload
        t = self.action_type

        if t == ActionType.CLASSIFY_INVOICE and isinstance(p, ClassifyInvoicePayload):
            return f"classify_invoice({p.invoice_id},{p.invoice_type},{p.hsn_code},{p.gst_slab})"
        elif t == ActionType.MATCH_ITC and isinstance(p, MatchITCPayload):
            return f"match_itc({p.purchase_invoice_id},{p.gstr2b_invoice_id})"
        elif t == ActionType.FLAG_DISCREPANCY and isinstance(p, FlagDiscrepancyPayload):
            return f"flag_discrepancy({p.invoice_id},{p.discrepancy_type},{p.recommended_action})"
        elif t == ActionType.ACCEPT_MISMATCH and isinstance(p, AcceptMismatchPayload):
            return f"accept_mismatch({p.purchase_invoice_id})"
        elif t == ActionType.DEFER_INVOICE and isinstance(p, DeferInvoicePayload):
            return f"defer_invoice({p.invoice_id})"
        elif t == ActionType.SET_SECTION_VALUE and isinstance(p, SetSectionValuePayload):
            return f"set_section({p.section},{p.value:.2f})"
        elif t == ActionType.GENERATE_RETURN and isinstance(p, GenerateReturnPayload):
            return f"generate_return({p.tax_period})"
        elif t == ActionType.SUBMIT_RETURN and isinstance(p, SubmitReturnPayload):
            return f"submit_return({p.tax_period})"
        elif t == ActionType.SKIP_INVOICE and isinstance(p, SkipInvoicePayload):
            return f"skip_invoice({p.invoice_id})"
        elif t == ActionType.REQUEST_CLARIFICATION and isinstance(p, RequestClarificationPayload):
            return f"request_clarification({p.invoice_id})"
        else:
            return str(t.value)
