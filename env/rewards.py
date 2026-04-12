from __future__ import annotations

from models.action import (
    GSTAction, ActionType,
    ClassifyInvoicePayload, MatchITCPayload,
    FlagDiscrepancyPayload, SetSectionValuePayload,
    DeferInvoicePayload,
)
from models.reward import RewardSignal


# ─────────────────────────────────────────────────────────────────────────────
# Potential-based shaping (guarantees policy invariance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_potential(state: dict) -> float:
    correct     = state.get("correct_decisions", 0)
    total       = max(state.get("total_decisions", 1), 1)
    matched_itc = state.get("matched_itc", 0.0)
    eligible    = max(state.get("eligible_itc", 1.0), 1.0)
    return 0.5 * (correct / total) + 0.5 * (matched_itc / eligible)


def shaped_reward(r_base: float, phi_s: float, phi_s_next: float, gamma: float = 0.99) -> float:
    """F(s, s') = γΦ(s') − Φ(s)  —  potential-based shaping."""
    return r_base + gamma * phi_s_next - phi_s


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reward_classify(action: ClassifyInvoicePayload, state: dict) -> tuple[float, dict, list[str]]:
    ground_truth = state.get("ground_truth", {}).get(action.invoice_id)
    if ground_truth is None:
        return -0.15, {"no_ground_truth": -0.15}, ["no_ground_truth"]

    sub: dict[str, float] = {}
    penalties: list[str] = []

    # GST slab (40%)
    if action.gst_slab == ground_truth.get("gst_slab"):
        sub["gst_slab"] = 0.40
    else:
        sub["gst_slab"] = -0.30
        penalties.append("wrong_slab")

    # HSN code (25%, partial 4-digit = 12.5%)
    pred_hsn = action.hsn_code or ""
    true_hsn = ground_truth.get("hsn_code", "")
    if pred_hsn == true_hsn:
        sub["hsn_code"] = 0.25
    elif pred_hsn[:4] == true_hsn[:4] and len(pred_hsn) >= 4:
        sub["hsn_code"] = 0.125
    else:
        sub["hsn_code"] = -0.20
        penalties.append("invalid_hsn")

    # Invoice type (15%)
    sub["invoice_type"] = 0.15 if action.invoice_type == ground_truth.get("invoice_type") else 0.0

    # ITC eligible (10%)
    sub["itc_eligible"] = 0.10 if action.itc_eligible == ground_truth.get("itc_eligible") else 0.0

    # Reverse charge (10%)
    sub["reverse_charge"] = 0.10 if action.reverse_charge == ground_truth.get("reverse_charge") else 0.0

    # Efficiency shaping bonus
    max_steps = state.get("max_steps", 50)
    steps_remaining = state.get("steps_remaining", 0)
    sub["efficiency"] = 0.05 * (steps_remaining / max(max_steps, 1))

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reward_match_itc(action: MatchITCPayload, state: dict) -> tuple[float, dict, list[str]]:
    # correct_matches: purchase_invoice_id → gstr2b_invoice_id (ground truth for Task 2)
    correct_matches = state.get("correct_matches", {})
    correct_gstr2b  = correct_matches.get(action.purchase_invoice_id)
    sub: dict[str, float] = {}
    penalties: list[str] = []

    if correct_gstr2b and correct_gstr2b == action.gstr2b_invoice_id:
        sub["match_correct"] = 0.50
    else:
        sub["match_correct"] = -0.40
        penalties.append("false_match")

    # Amount delta bonus: reward small delta as sign of genuine match
    purchase_map = state.get("purchase_map", {})
    gstr2b_map   = state.get("gstr2b_map", {})
    p_inv = purchase_map.get(action.purchase_invoice_id, {})
    g_inv = gstr2b_map.get(action.gstr2b_invoice_id, {})
    delta = abs(p_inv.get("taxable_value", 0) - g_inv.get("taxable_value", 0))

    if delta < 1000:
        sub["amount_delta"] = 0.20
    else:
        sub["amount_delta"] = 0.0

    # Penalty: matching invoices from different suppliers (false positive)
    p_gstin = p_inv.get("supplier_gstin")
    g_gstin = g_inv.get("supplier_gstin")
    if p_gstin and g_gstin and p_gstin != g_gstin:
        sub["different_supplier"] = -0.40
        penalties.append("false_match_different_supplier")

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties


def _reward_flag_discrepancy(action: FlagDiscrepancyPayload, state: dict) -> tuple[float, dict, list[str]]:
    ground_truth = state.get("discrepancy_truth", {}).get(action.invoice_id)
    sub: dict[str, float] = {}
    penalties: list[str] = []

    if ground_truth is None:
        sub["flag_unnecessary"] = -0.10
        penalties.append("unnecessary_flag")
        return -0.10, sub, penalties

    # Discrepancy type (15%)
    if action.discrepancy_type == ground_truth.get("discrepancy_type"):
        sub["discrepancy_type"] = 0.15
    else:
        sub["discrepancy_type"] = 0.0

    # Recommended action (15%)
    if action.recommended_action == ground_truth.get("recommended_action"):
        sub["recommended_action"] = 0.15
    else:
        sub["recommended_action"] = 0.0

    # Penalty: missed fraud flag
    if ground_truth.get("discrepancy_type") == "fake_invoice" and action.discrepancy_type != "fake_invoice":
        sub["missed_fraud"] = -0.30
        penalties.append("missed_fraud_flag")

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reward_set_section(action: SetSectionValuePayload, state: dict) -> tuple[float, dict, list[str]]:
    true_values: dict[str, float] = state.get("true_section_values", {})
    sub: dict[str, float] = {}
    penalties: list[str] = []

    true_val = true_values.get(action.section)
    if true_val is None:
        return 0.0, {}, []

    pct_error = abs(action.value - true_val) / (abs(true_val) + 1.0)

    if pct_error < 0.05:
        sub[action.section] = 0.20
    elif pct_error < 0.20:
        sub[action.section] = 0.10
    else:
        sub[action.section] = -0.10
        penalties.append(f"section_{action.section}_error_{pct_error:.2f}")

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties


def _reward_defer_invoice(action: DeferInvoicePayload, state: dict) -> tuple[float, dict, list[str]]:
    """
    Reward for DEFER_INVOICE action (Task 2).
    Correct when:  discrepancy_truth says correct_action_type == "defer_invoice"
                   (i.e. the invoice is not_in_2b from a high-compliance supplier)
    Penalise when: the invoice has an amount_diff or fake_invoice — should have been
                   flagged or matched, not deferred.
    """
    truth = state.get("discrepancy_truth", {}).get(action.invoice_id)
    sub: dict[str, float] = {}
    penalties: list[str] = []

    if truth is None:
        # Deferring an invoice that has no known issue → small penalty
        sub["unnecessary_defer"] = -0.10
        penalties.append("unnecessary_defer")
        return -0.10, sub, penalties

    if truth.get("correct_action_type") == "defer_invoice":
        # Correct: this is a not_in_2b invoice that should be deferred
        sub["correct_defer"] = 0.35
    elif truth.get("discrepancy_type") == "fake_invoice":
        # Wrong: should have flagged as fake, not deferred
        sub["deferred_fake_invoice"] = -0.40
        penalties.append("deferred_fake_invoice")
    elif truth.get("discrepancy_type") == "amount_diff":
        delta = abs(state.get("purchase_map", {}).get(action.invoice_id, {}).get("taxable_value", 0))
        if delta < 1000:
            sub["deferred_small_diff"] = -0.05   # should have accepted
            penalties.append("deferred_small_diff")
        else:
            sub["deferred_large_diff"] = -0.20   # should have disputed
            penalties.append("deferred_large_diff")
    else:
        sub["wrong_action"] = -0.10
        penalties.append("wrong_action_type")

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties


# ─────────────────────────────────────────────────────────────────────────────
# RewardEngine
# ─────────────────────────────────────────────────────────────────────────────

class RewardEngine:
    """Computes per-step rewards and applies potential-based shaping."""

    def compute(self, action: GSTAction, state: dict) -> RewardSignal:
        phi_s = compute_potential(state)

        # Duplicate-action penalty (loop detection)
        action_key = action.to_action_str()
        seen_actions: dict = state.setdefault("seen_actions", {})
        if action_key in seen_actions:
            seen_actions[action_key] += 1
            penalty = -0.20
            return RewardSignal(
                step_reward=penalty,
                cumulative_reward=state.get("cumulative_reward", 0.0) + penalty,
                sub_rewards={"loop_penalty": penalty},
                penalty_flags=["duplicate_action"],
                shaping_bonus=0.0,
            )
        seen_actions[action_key] = 1

        t = action.action_type
        p = action.payload

        if t == ActionType.CLASSIFY_INVOICE and isinstance(p, ClassifyInvoicePayload):
            base_reward, sub, flags = _reward_classify(p, state)
        elif t == ActionType.MATCH_ITC and isinstance(p, MatchITCPayload):
            base_reward, sub, flags = _reward_match_itc(p, state)
        elif t == ActionType.FLAG_DISCREPANCY and isinstance(p, FlagDiscrepancyPayload):
            base_reward, sub, flags = _reward_flag_discrepancy(p, state)
        elif t == ActionType.SET_SECTION_VALUE and isinstance(p, SetSectionValuePayload):
            base_reward, sub, flags = _reward_set_section(p, state)
        elif t == ActionType.DEFER_INVOICE and isinstance(p, DeferInvoicePayload):
            base_reward, sub, flags = _reward_defer_invoice(p, state)
        elif t == ActionType.SKIP_INVOICE:
            base_reward, sub, flags = -0.10, {"skip_penalty": -0.10}, ["skip_used"]
        elif t == ActionType.ACCEPT_MISMATCH:
            # Reward only if delta < 1000
            inv_id = getattr(p, "purchase_invoice_id", "")
            purchase_map = state.get("purchase_map", {})
            gstr2b_map   = state.get("gstr2b_map", {})
            p_inv = purchase_map.get(inv_id, {})
            g_match_id = state.get("best_match", {}).get(inv_id)
            g_inv = gstr2b_map.get(g_match_id, {}) if g_match_id else {}
            delta = abs(p_inv.get("taxable_value", 0) - g_inv.get("taxable_value", 0))
            if delta < 1000:
                base_reward, sub, flags = 0.20, {"accept_small_diff": 0.20}, []
            else:
                base_reward, sub, flags = -0.20, {"accepted_large_mismatch": -0.20}, ["accepted_large_mismatch"]
        elif t == ActionType.SUBMIT_RETURN:
            base_reward, sub, flags = _terminal_filing_reward(state)
        else:
            base_reward, sub, flags = 0.0, {}, []

        # Update state for potential computation
        if base_reward > 0:
            state["correct_decisions"] = state.get("correct_decisions", 0) + 1
        state["total_decisions"] = state.get("total_decisions", 0) + 1

        phi_s_next = compute_potential(state)
        shaped = shaped_reward(base_reward, phi_s, phi_s_next)
        shaped = round(min(1.0, max(-1.0, shaped)), 4)

        return RewardSignal(
            step_reward=shaped,
            cumulative_reward=state.get("cumulative_reward", 0.0) + shaped,
            sub_rewards=sub,
            penalty_flags=flags,
            shaping_bonus=round(shaped - base_reward, 4),
        )


def _terminal_filing_reward(state: dict) -> tuple[float, dict, list[str]]:
    """Terminal reward when agent submits GSTR-3B return."""
    true_vals: dict[str, float] = state.get("true_section_values", {})
    section_vals: dict[str, float] = state.get("section_values", {})
    sub: dict[str, float] = {}
    penalties: list[str] = []

    GSTR3B_SECTIONS = ["3.1a", "3.1b", "3.1c", "3.1d", "4a", "4b", "6.1", "6.2", "6.3"]

    all_within_5 = True
    section_scores = []
    for sec in GSTR3B_SECTIONS:
        agent_val = section_vals.get(sec, 0.0)
        true_val  = true_vals.get(sec, 0.0)
        pct_err   = abs(agent_val - true_val) / (abs(true_val) + 1.0)
        if pct_err >= 0.05:
            all_within_5 = False
        section_scores.append(max(0.0, 1.0 - pct_err / 0.05))

    base = sum(section_scores) / max(len(section_scores), 1)

    if all_within_5:
        sub["all_sections_5pct"] = 0.50
    else:
        sub["all_sections_5pct"] = 0.0

    # ITC offset check
    itc_ok = state.get("itc_offset_correct", False)
    sub["itc_offset"] = 0.25 if itc_ok else 0.0

    # Net payable within 2%
    agent_net = section_vals.get("net_payable", 0.0)
    true_net  = true_vals.get("net_payable", 0.0)
    net_err   = abs(agent_net - true_net) / (abs(true_net) + 1.0)
    sub["net_payable"] = 0.15 if net_err < 0.02 else 0.0

    # Step budget bonus
    steps_remaining = state.get("steps_remaining", 0)
    sub["step_budget"] = 0.10 if steps_remaining > 0 else 0.0

    # Penalty: unresolved discrepancies
    unresolved = state.get("unresolved_discrepancies", 0)
    if unresolved > 0:
        sub["unresolved_discrepancies"] = -0.50
        penalties.append("unresolved_discrepancies")

    # Penalty: underpayment > 2%
    if true_net > 0 and agent_net < true_net * 0.98:
        sub["underpayment"] = -0.30
        penalties.append("underpayment")

    # Penalty: ITC overclaim > 2% (episode-ending)
    agent_itc = state.get("total_itc_claimed", 0.0)
    true_itc  = state.get("eligible_itc", 0.0)
    if true_itc > 0 and agent_itc > true_itc * 1.02:
        sub["itc_overclaim"] = -1.00
        penalties.append("itc_overclaim")

    total = sum(sub.values())
    return round(min(1.0, max(-1.0, total)), 4), sub, penalties
