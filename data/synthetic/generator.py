from __future__ import annotations

import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from models.observation import InvoiceObservation, ITCMismatch, InvoiceStatus

# ─────────────────────────────────────────────────────────────────────────────
# Load lookup tables
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent
_HSN_TABLE_PATH = _DATA_DIR / "hsn_table.json"
_GST_RULES_PATH = _DATA_DIR.parent / "semantic" / "gst_rules.json"


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# GSTIN helpers
# ─────────────────────────────────────────────────────────────────────────────

STATE_CODES = [
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","20",
    "21","22","23","24","27","29","30","32","33","34","36",
]

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _gstin_checksum(gstin14: str) -> str:
    """Compute GSTIN check digit (mod-36 weighted sum)."""
    total = 0
    for i, ch in enumerate(gstin14):
        val = CHARSET.index(ch)
        total += val * (2 if i % 2 else 1)
        total = (total // 36) + (total % 36)
    check = (36 - (total % 36)) % 36
    return CHARSET[check]


def generate_gstin(rng: random.Random) -> str:
    state = rng.choice(STATE_CODES)
    pan   = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + \
            "".join(rng.choices("0123456789", k=4)) + \
            rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    entity = rng.choice("1234567890ABCDE")
    z = "Z"
    base = state + pan + entity + z
    check = _gstin_checksum(base)
    return base + check


def make_fake_gstin(rng: random.Random) -> str:
    """Return an invalid GSTIN (bad checksum) for fake-invoice edge case."""
    gstin = generate_gstin(rng)
    # Flip the last character
    last = gstin[-1]
    wrong = CHARSET[(CHARSET.index(last) + 1) % len(CHARSET)]
    return gstin[:-1] + wrong


# ─────────────────────────────────────────────────────────────────────────────
# SyntheticDataGenerator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Seeded synthetic invoice generator.

    Produces deterministic batches for Task 1, 2, and 3.
    Edge cases from PRD section 10 are injected at realistic prevalence rates.
    """

    BATCH_SIZES = {1: 20, 2: 30, 3: 1}   # invoices per task

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng  = random.Random(seed)
        self._hsn_table: list[dict] = []
        self._gst_rules: dict       = {}
        self._load_tables()

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_tables(self):
        try:
            self._hsn_table = _load_json(_HSN_TABLE_PATH)  # type: ignore[assignment]
        except FileNotFoundError:
            self._hsn_table = _FALLBACK_HSN_TABLE

        try:
            self._gst_rules = _load_json(_GST_RULES_PATH)  # type: ignore[assignment]
        except FileNotFoundError:
            self._gst_rules = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Invoice generation
    # ──────────────────────────────────────────────────────────────────────────

    def _random_invoice(
        self,
        buyer_gstin: str,
        tax_period: str,
        idx: int,
        force_fake: bool = False,
        force_missing_gstin: bool = False,
    ) -> tuple[InvoiceObservation, dict]:
        """Returns (InvoiceObservation, ground_truth_dict)."""
        rng = self._rng
        hsn_entry = rng.choice(self._hsn_table)
        hsn_code  = hsn_entry["hsn"]
        slab      = str(hsn_entry["slab"])
        supply    = hsn_entry.get("supply_type", "goods")
        desc      = hsn_entry.get("description", "Item")

        taxable = round(rng.uniform(5000, 200000), 2)

        # Determine inter vs intra state (GSTIN prefix check)
        buyer_state   = buyer_gstin[:2]
        supplier_gstin = generate_gstin(rng) if not force_fake else make_fake_gstin(rng)
        supplier_state = supplier_gstin[:2]
        inter_state    = (buyer_state != supplier_state)

        rate = int(slab) if slab.isdigit() else 0
        if inter_state:
            igst = round(taxable * rate / 100, 2)
            cgst = sgst = 0.0
        else:
            igst = 0.0
            cgst = sgst = round(taxable * rate / 200, 2)

        # Invoice type
        inv_type = rng.choices(
            ["B2B", "B2C", "EXPORT", "RCM"],
            weights=[0.60, 0.20, 0.10, 0.10],
        )[0]
        rcm = (inv_type == "RCM")
        itc = inv_type in ("B2B", "RCM") and not force_fake

        # Inject edge cases
        flags: list[str] = []
        if force_fake:
            flags.append("fake_invoice")
        if force_missing_gstin:
            supplier_gstin = None  # type: ignore[assignment]
            flags.append("missing_gstin")
            inv_type = "B2C"
            itc = False

        # Random date in tax_period
        year, month = map(int, tax_period.split("-"))
        day = rng.randint(1, 28)
        inv_date = date(year, month, day).isoformat()

        invoice_id = f"INV-{year}-{month:02d}-{idx:04d}"

        obs = InvoiceObservation(
            invoice_id=invoice_id,
            supplier_gstin=supplier_gstin,
            buyer_gstin=buyer_gstin,
            invoice_date=inv_date,
            invoice_type=inv_type,
            hsn_code=hsn_code,
            description=desc,
            taxable_value=taxable,
            igst_amount=igst,
            cgst_amount=cgst,
            sgst_amount=sgst,
            total_amount=round(taxable + igst + cgst + sgst, 2),
            gstr2b_match=None,
            status=InvoiceStatus.PENDING,
            flags=flags,
        )

        gt = {
            "invoice_id":     invoice_id,
            "invoice_type":   inv_type,
            "hsn_code":       hsn_code,
            "gst_slab":       slab,
            "supply_type":    supply,
            "itc_eligible":   itc,
            "reverse_charge": rcm,
        }
        return obs, gt

    # ──────────────────────────────────────────────────────────────────────────
    # Task 1 batch
    # ──────────────────────────────────────────────────────────────────────────

    def generate_task1_batch(self) -> dict:
        rng = self._rng
        buyer_gstin = generate_gstin(rng)
        tax_period  = f"{rng.randint(2023, 2024)}-{rng.randint(1, 12):02d}"
        n_invoices  = self.BATCH_SIZES[1]

        invoices:     list[InvoiceObservation] = []
        ground_truth: dict[str, dict]          = {}

        for i in range(n_invoices):
            force_fake    = rng.random() < 0.05   # 5%
            force_missing = rng.random() < 0.08   # 8%
            obs, gt = self._random_invoice(
                buyer_gstin, tax_period, i,
                force_fake=force_fake,
                force_missing_gstin=force_missing,
            )
            invoices.append(obs)
            ground_truth[obs.invoice_id] = gt

        supplier_profiles = self._generate_supplier_profiles(invoices, rng)

        return {
            "gstin":             buyer_gstin,
            "tax_period":        tax_period,
            "invoices":          invoices,
            "ground_truth":      ground_truth,
            "supplier_profiles": supplier_profiles,
            "eligible_itc":      sum(
                i.igst_amount + i.cgst_amount + i.sgst_amount
                for i in invoices
                if "fake_invoice" not in i.flags and "missing_gstin" not in i.flags
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Task 2 batch
    # ──────────────────────────────────────────────────────────────────────────

    def generate_task2_batch(self) -> dict:
        rng = self._rng
        buyer_gstin = generate_gstin(rng)
        tax_period  = f"{rng.randint(2023, 2024)}-{rng.randint(1, 12):02d}"
        n_invoices  = self.BATCH_SIZES[2]

        invoices, ground_truth = [], {}
        for i in range(n_invoices):
            obs, gt = self._random_invoice(buyer_gstin, tax_period, i)
            invoices.append(obs)
            ground_truth[obs.invoice_id] = gt

        # Build purchase_map and gstr2b_map with deliberate mismatches
        purchase_map:       dict[str, dict] = {}
        gstr2b_map:         dict[str, dict] = {}
        correct_matches:    dict[str, str]  = {}
        discrepancy_truth:  dict[str, dict] = {}
        best_match:         dict[str, str]  = {}
        mismatches:         list[ITCMismatch] = []

        year, month = map(int, tax_period.split("-"))

        for i, inv in enumerate(invoices):
            p_id = inv.invoice_id
            g_id = f"2B-{i:05d}"
            purchase_map[p_id] = {
                "invoice_id":    p_id,
                "invoice_number": p_id,
                "supplier_gstin": inv.supplier_gstin,
                "taxable_value":  inv.taxable_value,
                "igst_amount":    inv.igst_amount,
                "invoice_date":   inv.invoice_date,
                "hsn_code":       ground_truth[p_id]["hsn_code"],
            }

            roll = rng.random()
            if roll < 0.15:   # amount_diff < 1000
                g_taxable = inv.taxable_value - rng.uniform(100, 900)
                mismatch_type = "amount_diff"
                rec_action    = "accept_mismatch"
                delta = abs(inv.taxable_value - g_taxable)
            elif roll < 0.25: # amount_diff > 10K
                g_taxable = inv.taxable_value - rng.uniform(10000, 50000)
                mismatch_type = "amount_diff"
                rec_action    = "dispute"
                delta = abs(inv.taxable_value - g_taxable)
            elif roll < 0.28: # not_in_2b
                gstr2b_map[g_id] = {}
                mismatches.append(ITCMismatch(
                    purchase_invoice_id=p_id,
                    gstr2b_invoice_id=None,
                    mismatch_type="not_in_2b",
                    purchase_taxable=inv.taxable_value,
                    gstr2b_taxable=None,
                    delta=inv.taxable_value,
                ))
                # correct_action = DEFER_INVOICE (a separate action type, not flag_discrepancy)
                discrepancy_truth[p_id] = {
                    "discrepancy_type":   "not_in_2b",
                    "correct_action_type": "defer_invoice",   # ActionType.DEFER_INVOICE
                    "recommended_action": "defer",            # only used if agent flags instead of defers
                }
                continue
            elif roll < 0.31: # fake invoice
                g_taxable     = inv.taxable_value
                mismatch_type = "fake_invoice" if "fake_invoice" in inv.flags else "amount_diff"
                rec_action    = "dispute"
                delta = 0.0
            else:
                g_taxable = inv.taxable_value
                mismatch_type = None
                delta = 0.0

            gstr2b_map[g_id] = {
                "invoice_id":     g_id,
                "invoice_number": p_id,
                "supplier_gstin": inv.supplier_gstin,
                "taxable_value":  round(g_taxable, 2),
                "igst_amount":    inv.igst_amount,
                "invoice_date":   inv.invoice_date,
            }
            correct_matches[p_id] = g_id
            best_match[p_id]      = g_id

            if mismatch_type:
                mismatches.append(ITCMismatch(
                    purchase_invoice_id=p_id,
                    gstr2b_invoice_id=g_id,
                    mismatch_type=mismatch_type,
                    purchase_taxable=inv.taxable_value,
                    gstr2b_taxable=round(g_taxable, 2),
                    delta=round(delta, 2),
                ))
                if mismatch_type != "not_in_2b":
                    discrepancy_truth[p_id] = {
                        "discrepancy_type":   mismatch_type,
                        "recommended_action": rec_action,
                    }

        supplier_profiles = self._generate_supplier_profiles(invoices, rng)

        return {
            "gstin":              buyer_gstin,
            "tax_period":         tax_period,
            "invoices":           invoices,
            "purchase_map":       purchase_map,
            "gstr2b_map":         gstr2b_map,
            "correct_matches":    correct_matches,
            "discrepancy_truth":  discrepancy_truth,
            "best_match":         best_match,
            "mismatches":         mismatches,
            "ground_truth":       ground_truth,
            "supplier_profiles":  supplier_profiles,
            "eligible_itc":       sum(
                inv.igst_amount + inv.cgst_amount + inv.sgst_amount
                for inv in invoices
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Task 3 batch
    # ──────────────────────────────────────────────────────────────────────────

    def generate_task3_batch(self) -> dict:
        """Generate pre-reconciled aggregate values for GSTR-3B filing."""
        rng = self._rng
        buyer_gstin = generate_gstin(rng)
        tax_period  = f"{rng.randint(2023, 2024)}-{rng.randint(1, 12):02d}"

        taxable_outward = round(rng.uniform(500000, 5000000), 2)
        zero_rated      = round(rng.uniform(0, taxable_outward * 0.10), 2)
        exempted        = round(rng.uniform(0, taxable_outward * 0.05), 2)
        rcm_inward      = round(rng.uniform(0, 100000), 2)

        itc_igst  = round(rng.uniform(50000, 500000), 2)
        itc_ineli = round(itc_igst * rng.uniform(0.02, 0.10), 2)  # Section 17(5) reversals
        net_itc   = itc_igst - itc_ineli

        igst_liability = round(taxable_outward * 0.18, 2)
        cgst_liability = round(taxable_outward * 0.09, 2)
        sgst_liability = round(taxable_outward * 0.09, 2)

        # ITC offset: IGST ITC → IGST first, then CGST, then SGST
        igst_remaining = max(0.0, igst_liability - net_itc)
        overflow = max(0.0, net_itc - igst_liability)
        cgst_net = max(0.0, cgst_liability - overflow)
        overflow2 = max(0.0, overflow - cgst_liability)
        sgst_net = max(0.0, sgst_liability - overflow2)
        net_payable = round(igst_remaining + cgst_net + sgst_net, 2)

        true_section_values = {
            "3.1a": taxable_outward,
            "3.1b": zero_rated,
            "3.1c": exempted,
            "3.1d": rcm_inward,
            "4a":   itc_igst,
            "4b":   itc_ineli,
            "6.1":  igst_liability,
            "6.2":  cgst_liability,
            "6.3":  sgst_liability,
            "net_payable": net_payable,
            "total_itc":   net_itc,
        }

        return {
            "gstin":               buyer_gstin,
            "tax_period":          tax_period,
            "invoices":            [],
            "true_section_values": true_section_values,
            "eligible_itc":        net_itc,
            "mismatches":          [],
            "ground_truth":        {},
            "supplier_profiles":   {},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_supplier_profiles(
        self, invoices: list[InvoiceObservation], rng: random.Random
    ) -> dict[str, dict]:
        profiles = {}
        seen = set()
        for inv in invoices:
            gstin = inv.supplier_gstin
            if not gstin or gstin in seen:
                continue
            seen.add(gstin)
            profiles[gstin] = {
                "compliance_rate": round(rng.uniform(0.80, 1.00), 4),
                "avg_delay_days":  round(rng.uniform(0, 15), 1),
                "cancelled_pct":   round(rng.uniform(0, 0.05), 4),
            }
        return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Fallback HSN table (used if hsn_table.json is missing)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_HSN_TABLE = [
    {"hsn": "8471", "description": "Laptop Computer",           "slab": "18", "supply_type": "goods"},
    {"hsn": "6109", "description": "T-Shirts",                  "slab": "5",  "supply_type": "goods"},
    {"hsn": "9983", "description": "IT Consulting Services",    "slab": "18", "supply_type": "services"},
    {"hsn": "3004", "description": "Pharmaceutical Products",   "slab": "12", "supply_type": "goods"},
    {"hsn": "8528", "description": "Television Monitors",       "slab": "28", "supply_type": "goods"},
    {"hsn": "0401", "description": "Milk and Cream",            "slab": "0",  "supply_type": "goods"},
    {"hsn": "4901", "description": "Printed Books",             "slab": "0",  "supply_type": "goods"},
    {"hsn": "9954", "description": "Construction Services",     "slab": "18", "supply_type": "services"},
    {"hsn": "8517", "description": "Mobile Phones",             "slab": "18", "supply_type": "goods"},
    {"hsn": "2710", "description": "Petroleum Oils",            "slab": "18", "supply_type": "goods"},
    {"hsn": "6203", "description": "Men's Suits and Blazers",   "slab": "12", "supply_type": "goods"},
    {"hsn": "8443", "description": "Printing Machinery",        "slab": "18", "supply_type": "goods"},
    {"hsn": "9985", "description": "Support Services",          "slab": "18", "supply_type": "services"},
    {"hsn": "7208", "description": "Flat-Rolled Steel",         "slab": "18", "supply_type": "goods"},
    {"hsn": "3926", "description": "Plastic Articles",          "slab": "18", "supply_type": "goods"},
]
