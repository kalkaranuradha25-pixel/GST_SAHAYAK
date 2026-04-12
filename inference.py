"""
inference.py — Hackathon Entry Point (MANDATORY)

Output protocol (must be exact — any deviation causes scoring failure):
    [START] task=<name> env=gst-intelligence model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Rules:
    - One [START] at episode begin
    - One [STEP] per step, immediately after env.step() returns
    - One [END] after env.close(), ALWAYS (even on exception) — use try/finally
    - reward and rewards formatted to 2 decimal places
    - done and success are lowercase: true / false
    - error is raw last_action_error string, or null
    - All fields on a SINGLE line, no embedded newlines
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Environment variables (MANDATORY — do not change defaults or remove checks)
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")   # MUST have default
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")                # MUST have default
HF_TOKEN     = os.getenv("HF_TOKEN")                                     # NO default — mandatory

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ─────────────────────────────────────────────────────────────────────────────
# Imports (after env var check)
# ─────────────────────────────────────────────────────────────────────────────

import openai

from env.gst_env import GSTEnvironment
from models.action import (
    GSTAction, ActionType,
    ClassifyInvoicePayload, MatchITCPayload,
    FlagDiscrepancyPayload, AcceptMismatchPayload,
    DeferInvoicePayload, SetSectionValuePayload,
    GenerateReturnPayload, SubmitReturnPayload,
    SkipInvoicePayload,
)
from models.observation import GSTObservation

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (MANDATORY — no direct HTTP)
# ─────────────────────────────────────────────────────────────────────────────

client = openai.OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Task metadata
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAMES = {
    1: "invoice_classification",
    2: "itc_reconciliation",
    3: "gstr3b_filing",
}

MAX_STEPS = {1: 50, 2: 150, 3: 300}

GSTR3B_SECTION_ORDER = [
    "3.1a", "3.1b", "3.1c", "3.1d",
    "4a", "4b",
    "6.1", "6.2", "6.3",
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Indian GST compliance agent.
You process invoices and GST returns for Indian businesses.
You always respond with a single valid JSON object — no markdown, no explanation.
Respond only with the JSON action object requested."""


def _obs_summary(obs: GSTObservation) -> str:
    """Compact observation string for the LLM prompt."""
    inv = obs.current_invoice
    inv_str = "none"
    if inv:
        inv_str = (
            f"id={inv.invoice_id} type={inv.invoice_type} "
            f"hsn={inv.hsn_code} taxable={inv.taxable_value:.2f} "
            f"flags={inv.flags}"
        )

    mismatches_str = ""
    if obs.mismatches:
        top = obs.mismatches[0]
        mismatches_str = (
            f"\nTop mismatch: purchase={top.purchase_invoice_id} "
            f"type={top.mismatch_type} delta={top.delta:.2f}"
        )

    gstr = obs.gstr3b
    sections_done = gstr.sections_completed

    return (
        f"task={obs.task_id} step={obs.step_number} pending={obs.pending_count} "
        f"flagged={obs.flagged_count} cumulative_reward={obs.cumulative_reward:.2f}\n"
        f"current_invoice: {inv_str}{mismatches_str}\n"
        f"matched_itc={obs.matched_itc_amount:.2f} disputed={obs.disputed_itc_amount:.2f}\n"
        f"gstr3b_sections_done={sections_done}\n"
        f"steps_remaining={obs.steps_remaining}"
    )


def _task1_prompt(obs: GSTObservation) -> str:
    inv = obs.current_invoice
    if not inv:
        return '{"action_type":"skip_invoice","invoice_id":"none","reason":"no invoice"}'

    similar = ""
    if obs.similar_past_invoices:
        similar = f"\nSimilar past: {obs.similar_past_invoices[:2]}"

    supplier_profile = ""
    if obs.known_supplier_profile:
        p = obs.known_supplier_profile
        supplier_profile = (
            f"\nSupplier compliance rate: {p.get('historical_compliance_rate', 'unknown')}"
        )

    return f"""Classify this GST invoice. Respond with JSON only.

Invoice:
  id: {inv.invoice_id}
  type hint: {inv.invoice_type}
  description: {inv.description}
  hsn_code: {inv.hsn_code}
  taxable_value: {inv.taxable_value}
  igst: {inv.igst_amount}  cgst: {inv.cgst_amount}  sgst: {inv.sgst_amount}
  supplier_gstin: {inv.supplier_gstin}
  flags: {inv.flags}
{similar}{supplier_profile}

Respond with exactly this JSON (fill in values):
{{
  "action_type": "classify_invoice",
  "invoice_id": "{inv.invoice_id}",
  "invoice_type": "<B2B|B2C|EXPORT|RCM|ISD|EXEMPT>",
  "hsn_code": "<4-8 digit code>",
  "gst_slab": "<0|5|12|18|28|exempt>",
  "supply_type": "<goods|services>",
  "itc_eligible": <true|false>,
  "reverse_charge": <true|false>,
  "reasoning": "<one line>"
}}"""


def _task2_prompt(obs: GSTObservation) -> str:
    if not obs.mismatches:
        return '{"action_type":"skip_invoice","invoice_id":"done","reason":"no mismatches"}'

    top = obs.mismatches[0]
    inv_id = top.purchase_invoice_id
    g_id   = top.gstr2b_invoice_id

    return f"""Reconcile this ITC mismatch. Respond with JSON only.

Mismatch:
  purchase_id: {inv_id}
  gstr2b_id: {g_id}
  mismatch_type: {top.mismatch_type}
  purchase_taxable: {top.purchase_taxable}
  gstr2b_taxable: {top.gstr2b_taxable}
  delta: {top.delta}

Rules:
- not_in_2b → use defer_invoice action
- amount_diff < 1000 → accept_mismatch
- amount_diff > 10000 → flag_discrepancy with recommended_action=dispute
- fake_invoice → flag_discrepancy with recommended_action=dispute
- cancelled → flag_discrepancy with recommended_action=hold_itc
- duplicate → flag_discrepancy with recommended_action=write_off
- If GSTIN matches and delta is small → match_itc

Choose ONE action. Respond with exactly ONE of these JSON formats:

For match_itc:
{{"action_type":"match_itc","purchase_invoice_id":"{inv_id}","gstr2b_invoice_id":"{g_id}","confidence":0.95}}

For flag_discrepancy:
{{"action_type":"flag_discrepancy","invoice_id":"{inv_id}","discrepancy_type":"<type>","recommended_action":"<hold_itc|defer|dispute|write_off>","notes":"<reason>"}}

For accept_mismatch:
{{"action_type":"accept_mismatch","purchase_invoice_id":"{inv_id}","reason":"delta < 1000"}}

For defer_invoice:
{{"action_type":"defer_invoice","invoice_id":"{inv_id}","reason":"not in GSTR-2B, await supplier filing"}}"""


def _task3_prompt(obs: GSTObservation, section: str) -> str:
    gstr = obs.gstr3b
    sections_done = gstr.sections_completed

    section_descriptions = {
        "3.1a": "Total outward taxable supplies (excluding zero-rated and exempt)",
        "3.1b": "Zero-rated outward supplies",
        "3.1c": "Exempt outward supplies",
        "3.1d": "Inward supplies liable to reverse charge",
        "4a":   "ITC available — IGST (Input Tax Credit)",
        "4b":   "ITC to be reversed — Section 17(5) ineligible items",
        "6.1":  "IGST tax payable on outward supplies",
        "6.2":  "CGST tax payable on outward supplies",
        "6.3":  "SGST tax payable on outward supplies",
    }

    return f"""Fill GSTR-3B section {section}. Respond with JSON only.

Section: {section} — {section_descriptions.get(section, '')}

Current GSTR-3B state:
  taxable_outward: {gstr.taxable_outward}
  zero_rated: {gstr.zero_rated}
  exempted: {gstr.exempted}
  rcm_inward: {gstr.rcm_inward}
  itc_igst: {gstr.itc_igst}
  itc_ineligible: {gstr.itc_ineligible}
  igst_payable: {gstr.igst_payable}
  cgst_payable: {gstr.cgst_payable}
  sgst_payable: {gstr.sgst_payable}
  sections_completed: {sections_done}

Matched ITC: {obs.matched_itc_amount:.2f}
Steps remaining: {obs.steps_remaining}

ITC offset rules (legally mandated):
  IGST ITC → offsets IGST first, then CGST, then SGST
  CGST ITC → offsets CGST only
  SGST ITC → offsets SGST only

Respond with exactly:
{{"action_type":"set_section_value","section":"{section}","value":<float>}}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call + action parser
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, task_id: int) -> dict:
    """Call the LLM and return the parsed JSON action dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _parse_action(data: dict, obs: GSTObservation) -> Optional[GSTAction]:
    """Convert raw JSON dict from LLM into a typed GSTAction."""
    ts  = datetime.now(timezone.utc).isoformat()
    act = data.get("action_type", "")
    reasoning = data.get("reasoning")

    try:
        if act == "classify_invoice":
            payload = ClassifyInvoicePayload(
                invoice_id=data["invoice_id"],
                invoice_type=data["invoice_type"],
                hsn_code=str(data["hsn_code"]),
                gst_slab=str(data["gst_slab"]),
                supply_type=data["supply_type"],
                itc_eligible=bool(data["itc_eligible"]),
                reverse_charge=bool(data["reverse_charge"]),
            )
        elif act == "match_itc":
            payload = MatchITCPayload(
                purchase_invoice_id=data["purchase_invoice_id"],
                gstr2b_invoice_id=data["gstr2b_invoice_id"],
                confidence=float(data.get("confidence", 0.9)),
            )
        elif act == "flag_discrepancy":
            payload = FlagDiscrepancyPayload(
                invoice_id=data["invoice_id"],
                discrepancy_type=data["discrepancy_type"],
                recommended_action=data["recommended_action"],
                notes=str(data.get("notes", "")),
            )
        elif act == "accept_mismatch":
            payload = AcceptMismatchPayload(
                purchase_invoice_id=data["purchase_invoice_id"],
                reason=str(data.get("reason", "")),
            )
        elif act == "defer_invoice":
            payload = DeferInvoicePayload(
                invoice_id=data.get("invoice_id", ""),
                reason=str(data.get("reason", "")),
            )
        elif act == "set_section_value":
            payload = SetSectionValuePayload(
                section=data["section"],
                value=float(data["value"]),
            )
        elif act == "generate_return":
            payload = GenerateReturnPayload(tax_period=data.get("tax_period", ""))
        elif act == "submit_return":
            payload = SubmitReturnPayload(
                tax_period=data.get("tax_period", ""),
                declaration=bool(data.get("declaration", True)),
            )
        elif act == "skip_invoice":
            inv_id = (
                data.get("invoice_id")
                or (obs.current_invoice.invoice_id if obs.current_invoice else "unknown")
            )
            payload = SkipInvoicePayload(invoice_id=inv_id, reason=data.get("reason"))
        else:
            return None

        return GSTAction(
            action_type=ActionType(act),
            payload=payload,
            timestamp=ts,
            agent_reasoning=reasoning,
        )
    except (KeyError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fallback actions (used when LLM parse fails)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_action(obs: GSTObservation, task_id: int) -> GSTAction:
    """Deterministic fallback when LLM returns unparseable output."""
    ts = datetime.now(timezone.utc).isoformat()

    if task_id == 1:
        inv = obs.current_invoice
        if inv:
            payload = ClassifyInvoicePayload(
                invoice_id=inv.invoice_id,
                invoice_type=inv.invoice_type or "B2B",
                hsn_code=inv.hsn_code or "9983",
                gst_slab="18",
                supply_type="goods",
                itc_eligible=True,
                reverse_charge=False,
            )
            return GSTAction(action_type=ActionType.CLASSIFY_INVOICE, payload=payload, timestamp=ts)
        inv_id = "unknown"
        return GSTAction(
            action_type=ActionType.SKIP_INVOICE,
            payload=SkipInvoicePayload(invoice_id=inv_id, reason="fallback"),
            timestamp=ts,
        )

    elif task_id == 2:
        if obs.mismatches:
            top = obs.mismatches[0]
            if top.mismatch_type == "not_in_2b":
                return GSTAction(
                    action_type=ActionType.DEFER_INVOICE,
                    payload=DeferInvoicePayload(invoice_id=top.purchase_invoice_id, reason="not in 2B"),
                    timestamp=ts,
                )
            if top.delta < 1000:
                return GSTAction(
                    action_type=ActionType.ACCEPT_MISMATCH,
                    payload=AcceptMismatchPayload(purchase_invoice_id=top.purchase_invoice_id, reason="small delta"),
                    timestamp=ts,
                )
            return GSTAction(
                action_type=ActionType.FLAG_DISCREPANCY,
                payload=FlagDiscrepancyPayload(
                    invoice_id=top.purchase_invoice_id,
                    discrepancy_type=top.mismatch_type,
                    recommended_action="dispute",
                    notes="fallback",
                ),
                timestamp=ts,
            )
        return GSTAction(
            action_type=ActionType.SKIP_INVOICE,
            payload=SkipInvoicePayload(invoice_id="done", reason="no mismatches"),
            timestamp=ts,
        )

    else:  # task 3
        sections_done = obs.gstr3b.sections_completed
        for sec in GSTR3B_SECTION_ORDER:
            if sec not in sections_done:
                return GSTAction(
                    action_type=ActionType.SET_SECTION_VALUE,
                    payload=SetSectionValuePayload(section=sec, value=0.0),
                    timestamp=ts,
                )
        if len(sections_done) >= len(GSTR3B_SECTION_ORDER):
            return GSTAction(
                action_type=ActionType.GENERATE_RETURN,
                payload=GenerateReturnPayload(tax_period=obs.tax_period),
                timestamp=ts,
            )
        return GSTAction(
            action_type=ActionType.SUBMIT_RETURN,
            payload=SubmitReturnPayload(tax_period=obs.tax_period, declaration=True),
            timestamp=ts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_id: int = 1, seed: int = 42) -> bool:
    """
    Run one full episode for the given task.
    Returns True if episode completed successfully.

    Emits [START], [STEP]×n, [END] to stdout.
    """
    task_name = TASK_NAMES[task_id]
    env = GSTEnvironment()
    rewards_log: list[float] = []
    steps = 0
    success = False

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env=gst-intelligence model={MODEL_NAME}", flush=True)

    try:
        obs, _ = env.reset(seed=seed, task_id=task_id)
        max_steps = MAX_STEPS[task_id]

        # Track filing section progress for task 3
        sections_submitted: list[str] = []
        generated = False

        for step_n in range(1, max_steps + 1):
            steps = step_n
            error_str = "null"

            # ── Choose action ─────────────────────────────────────────────
            try:
                if task_id == 1:
                    prompt = _task1_prompt(obs)
                elif task_id == 2:
                    prompt = _task2_prompt(obs)
                else:
                    # Task 3: work through sections in order
                    sections_done = obs.gstr3b.sections_completed
                    remaining_sections = [s for s in GSTR3B_SECTION_ORDER if s not in sections_done]

                    if remaining_sections:
                        next_sec = remaining_sections[0]
                        prompt = _task3_prompt(obs, next_sec)
                    elif not generated:
                        prompt = f'{{"action_type":"generate_return","tax_period":"{obs.tax_period}"}}'
                    else:
                        prompt = f'{{"action_type":"submit_return","tax_period":"{obs.tax_period}","declaration":true}}'

                raw_data = _call_llm(prompt, task_id)
                action   = _parse_action(raw_data, obs)
                if action is None:
                    action = _fallback_action(obs, task_id)

            except Exception:
                action = _fallback_action(obs, task_id)

            # ── env.step() ───────────────────────────────────────────────
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_log.append(reward)

            last_error = info.get("last_action_error")
            error_str  = last_error if last_error else "null"
            done_flag  = "true" if (terminated or truncated) else "false"

            # ── [STEP] ───────────────────────────────────────────────────
            action_str = action.to_action_str()
            print(
                f"[STEP] step={step_n} action={action_str} "
                f"reward={reward:.2f} done={done_flag} error={error_str}",
                flush=True,
            )

            # Track generate_return for task 3 sequencing
            if action.action_type == ActionType.GENERATE_RETURN:
                generated = True

            if terminated or truncated:
                success = terminated and not reward_signal_failed(rewards_log)
                break

        # Loop exhausted all steps without a terminal signal — episode failed
        # (success remains False from initialisation)

    except Exception:
        success = False  # [STEP] is NOT emitted here — only after env.step()

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_log)
        print(
            f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
            flush=True,
        )

    return success


def reward_signal_failed(rewards: list[float]) -> bool:
    """Heuristic: episode failed if cumulative reward is strongly negative."""
    return sum(rewards) < -2.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Entry point for `gst-inference` CLI (defined in pyproject.toml)."""
    import argparse

    parser = argparse.ArgumentParser(description="GST Intelligence RL — Inference")
    parser.add_argument("--task",  type=int, default=1, choices=[1, 2, 3], help="Task ID")
    parser.add_argument("--seed",  type=int, default=42,                   help="Episode seed")
    args = parser.parse_args()

    run_episode(task_id=args.task, seed=args.seed)


if __name__ == "__main__":
    main()
