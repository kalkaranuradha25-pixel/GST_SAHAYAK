from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class InvoiceStatus(str, Enum):
    PENDING    = "pending"
    CLASSIFIED = "classified"
    MATCHED    = "matched"
    FLAGGED    = "flagged"
    DISPUTED   = "disputed"


class InvoiceObservation(BaseModel):
    invoice_id:       str
    supplier_gstin:   Optional[str] = None   # None = missing (red flag)
    buyer_gstin:      str
    invoice_date:     str                    # "YYYY-MM-DD"
    invoice_type:     str                    # B2B / B2C / EXPORT / RCM / ISD
    hsn_code:         Optional[str] = None   # 4-8 digit HSN or SAC
    description:      str
    taxable_value:    float
    igst_amount:      float
    cgst_amount:      float
    sgst_amount:      float
    total_amount:     float
    gstr2b_match:     Optional[str] = None   # Matched GSTR-2B entry ID or None
    status:           InvoiceStatus = InvoiceStatus.PENDING
    flags:            list[str] = Field(default_factory=list)
    # ["missing_gstin","rate_mismatch","duplicate","fake_invoice","not_in_2b"]


class ITCMismatch(BaseModel):
    purchase_invoice_id: str
    gstr2b_invoice_id:   Optional[str] = None
    mismatch_type:       str  # amount_diff / gstin_missing / not_in_2b / rate_mismatch / cancelled / duplicate
    purchase_taxable:    float
    gstr2b_taxable:      Optional[float] = None
    delta:               float


class GSTR3BSummary(BaseModel):
    taxable_outward:    float = 0.0   # 3.1(a)
    zero_rated:         float = 0.0   # 3.1(b)
    exempted:           float = 0.0   # 3.1(c)
    rcm_inward:         float = 0.0   # 3.1(d)
    itc_igst:           float = 0.0   # 4(A)(5)
    itc_cgst:           float = 0.0
    itc_sgst:           float = 0.0
    itc_ineligible:     float = 0.0   # 4(B) reversals
    igst_payable:       float = 0.0   # 6.1
    cgst_payable:       float = 0.0   # 6.2
    sgst_payable:       float = 0.0   # 6.3
    net_payable:        float = 0.0
    sections_completed: list[str] = Field(default_factory=list)


class GSTObservation(BaseModel):
    episode_id:              str
    task_id:                 int                       # 1, 2, or 3
    step_number:             int
    gstin:                   str
    tax_period:              str                       # "2024-03"
    current_invoice:         Optional[InvoiceObservation] = None
    total_invoices:          int
    classified_count:        int
    pending_count:           int
    flagged_count:           int
    mismatches:              list[ITCMismatch] = Field(default_factory=list)
    matched_itc_amount:      float
    disputed_itc_amount:     float
    gstr3b:                  GSTR3BSummary = Field(default_factory=GSTR3BSummary)
    steps_remaining:         int
    cumulative_reward:       float
    last_action_result:      Optional[str] = None   # "success"|"invalid"|"penalty"
    similar_past_invoices:   list[dict] = Field(default_factory=list)
    known_supplier_profile:  Optional[dict] = None
