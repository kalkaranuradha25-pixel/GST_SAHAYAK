"""
app.py — GST Intelligence RL — Gradio UI v2
Dashboard: left sidebar (controls + live stats) + right main panel (invoice cards + log)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Generator

import gradio as gr
import pandas as pd

from env.gst_env import GSTEnvironment
from models.observation import GSTObservation
from agent.baseline_agent import BaselineAgent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_INFO = {
    1: {
        "name": "Invoice Classifier",
        "full_name": "Invoice Classification",
        "difficulty": "Easy",
        "difficulty_color": "#22c55e",
        "max_steps": 50,
        "accent": "#3b82f6",
        "description": "Classify each invoice: type (B2B/B2C/RCM…), HSN code, GST slab, ITC eligibility, and reverse-charge flag.",
        "reward_tips": "+0.40 correct slab · +0.25 correct HSN · −0.30 wrong slab",
    },
    2: {
        "name": "ITC Reconciliation",
        "full_name": "ITC Reconciliation",
        "difficulty": "Medium",
        "difficulty_color": "#f59e0b",
        "max_steps": 150,
        "accent": "#f59e0b",
        "description": "Match purchase register invoices against GSTR-2B. Detect amount mismatches, fake invoices, cancelled entries, and missing filings.",
        "reward_tips": "+0.50 correct match · −0.40 false match · −0.30 missed fraud",
    },
    3: {
        "name": "GSTR-3B Filing",
        "full_name": "GSTR-3B Filing",
        "difficulty": "Hard",
        "difficulty_color": "#ef4444",
        "max_steps": 300,
        "accent": "#a855f7",
        "description": "Fill all 9 GSTR-3B sections in legal order, apply ITC offset rules (IGST→IGST/CGST/SGST), and submit the return within budget.",
        "reward_tips": "+0.20 section within 5% · −1.00 ITC overclaim (episode ends)",
    },
}

ACTION_ICONS = {
    "classify_invoice":      "🏷",
    "match_itc":             "🔗",
    "flag_discrepancy":      "🚩",
    "accept_mismatch":       "✅",
    "defer_invoice":         "⏳",
    "set_section_value":     "📝",
    "generate_return":       "⚙",
    "submit_return":         "📤",
    "skip_invoice":          "⏭",
    "request_clarification": "❓",
}

SECTION_INFO = {
    "3.1a": "Outward taxable supplies",
    "3.1b": "Zero-rated outward",
    "3.1c": "Exempt outward",
    "3.1d": "RCM inward supplies",
    "4a":   "ITC available (IGST)",
    "4b":   "ITC ineligible §17(5)",
    "6.1":  "IGST payable",
    "6.2":  "CGST payable",
    "6.3":  "SGST payable",
}

STATUS_STYLE = {
    "pending":    ("#f59e0b", "#f59e0b1a"),
    "classified": ("#22c55e", "#22c55e1a"),
    "flagged":    ("#ef4444", "#ef44441a"),
    "deferred":   ("#a855f7", "#a855f71a"),
    "skipped":    ("#64748b", "#64748b1a"),
    "matched":    ("#22c55e", "#22c55e1a"),
    "disputed":   ("#ef4444", "#ef44441a"),
    "accepted":   ("#10b981", "#10b9811a"),
    "submitted":  ("#3b82f6", "#3b82f61a"),
    "done":       ("#22c55e", "#22c55e1a"),
}

MISMATCH_TYPE_COLOR = {
    "amount_diff":  "#f59e0b",
    "fake_invoice": "#ef4444",
    "cancelled":    "#a855f7",
    "duplicate":    "#f97316",
    "not_in_2b":    "#3b82f6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Episode-level state (single-user hackathon demo — global is fine)
# ─────────────────────────────────────────────────────────────────────────────

class _EpState:
    def __init__(self):
        self.env: GSTEnvironment | None = None
        self.obs: GSTObservation | None = None
        self.task_id = 1
        self.seed = 42
        self.step = 0
        self.max_steps = 50
        self.rewards: list[float] = []
        self.log_entries: list[str] = []          # rendered HTML entries
        self.invoice_map: dict[str, dict] = {}    # id → {inv, status, action, reward}
        self.done = False

_S = _EpState()

# ─────────────────────────────────────────────────────────────────────────────
# HTML component renderers
# ─────────────────────────────────────────────────────────────────────────────

def _badge(label: str, color: str, bg: str) -> str:
    return (
        f'<span style="background:{bg}; color:{color}; border:1px solid {color}40;'
        f' padding:2px 9px; border-radius:20px; font-size:0.62rem; font-weight:700;'
        f' letter-spacing:0.06em; text-transform:uppercase;">{label}</span>'
    )


def render_banner(task_id: int, step: int, done: bool, error: str = "") -> str:
    total = sum(_S.rewards)
    if error:
        c, icon, msg = "#ef4444", "✗", f"Error — {error}"
    elif done:
        c = "#22c55e" if total > 0 else "#f59e0b"
        icon, msg = "✓", f"Episode complete · Steps: {step} · Total reward: {total:+.2f}"
    elif step == 0:
        c, icon, msg = "#3b82f6", "●", f"Ready — Task: {TASK_INFO[task_id]['name']}"
    else:
        c, icon, msg = "#22c55e", "◉", f"Running — {TASK_INFO[task_id]['name']} · Step {step}"

    anim = "animation:pulse 2s infinite;" if (not done and step > 0) else ""
    return f"""
<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}</style>
<div style="background:{c}12; border-left:4px solid {c}; padding:12px 20px;
     border-radius:0 8px 8px 0; display:flex; align-items:center; gap:10px; margin-bottom:2px;">
  <span style="color:{c}; font-size:1rem; {anim}">{icon}</span>
  <span style="color:{c}; font-weight:600; font-size:0.9rem;">{msg}</span>
</div>"""


def render_stats(step: int, max_steps: int, rewards: list, pending: int) -> str:
    total = sum(rewards)
    pct = int(step / max_steps * 100) if max_steps > 0 else 0
    r_col = "#22c55e" if total >= 0 else "#ef4444"
    p_col = "#f59e0b" if pending > 0 else "#22c55e"
    bar_col = "#3b82f6" if pct < 80 else "#22c55e"

    def _box(val, label, col="#e2e8f0"):
        return (
            f'<div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px;'
            f' padding:10px 6px; text-align:center;">'
            f'<div style="font-size:1.3rem; font-weight:700; color:{col};">{val}</div>'
            f'<div style="font-size:0.6rem; color:#475569; text-transform:uppercase;'
            f' letter-spacing:0.08em; margin-top:2px;">{label}</div>'
            f'</div>'
        )

    return f"""
<div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:2px 0 8px;">
    {_box(step, "Step")}
    {_box(max_steps, "Max Steps")}
    {_box(f"{total:+.2f}", "Cum. Reward", r_col)}
    {_box(pending, "Pending", p_col)}
  </div>
  <div>
    <div style="display:flex; justify-content:space-between; font-size:0.65rem;
         color:#475569; margin-bottom:4px;">
      <span>Progress</span><span>{pct}%</span>
    </div>
    <div style="background:#0f172a; border-radius:4px; height:5px; overflow:hidden;">
      <div style="background:{bar_col}; height:100%; width:{pct}%;
           border-radius:4px; transition:width .4s ease;"></div>
    </div>
  </div>
</div>"""


def render_context(obs: GSTObservation) -> str:
    gstr = obs.gstr3b
    filing = f"{len(gstr.sections_completed)}/9 sections" if obs.task_id == 3 else "n/a"
    itc    = f"₹{obs.matched_itc_amount:,.0f}"
    liab   = f"₹{(gstr.igst_payable + gstr.cgst_payable + gstr.sgst_payable):,.0f}"

    rows = [
        ("GSTIN",   (obs.gstin[:14] + "…") if len(obs.gstin) > 15 else obs.gstin),
        ("Period",  obs.tax_period),
        ("Filing",  filing),
        ("ITC",     itc),
        ("Liability", liab),
    ]
    inner = "".join(
        f'<div style="display:flex; justify-content:space-between; padding:5px 0;'
        f' border-bottom:1px solid #0f172a;">'
        f'<span style="color:#475569; font-size:0.75rem;">{k}</span>'
        f'<span style="color:#cbd5e1; font-size:0.75rem; font-weight:500;">{v}</span>'
        f'</div>'
        for k, v in rows
    )
    return f'<div style="padding:2px 0;">{inner}</div>'


def render_invoices(obs: GSTObservation) -> str:
    if not _S.invoice_map:
        return (
            '<div style="color:#475569; text-align:center; padding:50px 20px;">'
            '<div style="font-size:2rem; margin-bottom:8px;">📄</div>'
            '<div style="font-size:0.85rem;">Press <b>Reset / New Episode</b> to load invoices</div>'
            '</div>'
        )

    cur_id = obs.current_invoice.invoice_id if obs.current_invoice else None
    cards = []

    for inv_id, entry in _S.invoice_map.items():
        inv     = entry["inv"]
        status  = entry.get("status", "pending")
        act_str = entry.get("action", "")
        r_val   = entry.get("reward", 0.0)
        is_cur  = inv_id == cur_id

        sc, bg = STATUS_STYLE.get(status, ("#94a3b8", "#94a3b81a"))
        border  = "#3b82f6" if is_cur else "#1e293b"
        row_bg  = "#162032" if is_cur else "#131c2e"

        # tax breakdown
        tax_parts = []
        if inv.igst_amount: tax_parts.append(f"IGST ₹{inv.igst_amount:,.0f}")
        if inv.cgst_amount: tax_parts.append(f"CGST ₹{inv.cgst_amount:,.0f}")
        if inv.sgst_amount: tax_parts.append(f"SGST ₹{inv.sgst_amount:,.0f}")
        tax_str = " · ".join(tax_parts) or "—"

        # flags
        flag_html = "".join(
            f'<span style="background:#ef444420; color:#ef4444; border-radius:4px;'
            f' padding:1px 6px; font-size:0.62rem; margin-right:3px;">{f}</span>'
            for f in (inv.flags or [])
        )

        # action taken
        action_html = ""
        if act_str:
            rc = "#22c55e" if r_val >= 0 else "#ef4444"
            action_html = (
                f'<div style="font-size:0.7rem; color:#475569; margin-top:5px;">'
                f'→ <span style="color:#94a3b8;">{act_str}</span>'
                f' <span style="color:{rc};">({r_val:+.2f})</span></div>'
            )

        cur_tag = (
            '<span style="background:#3b82f620; color:#3b82f6; border-radius:4px;'
            ' padding:1px 7px; font-size:0.62rem; font-weight:600; margin-left:6px;">CURRENT</span>'
            if is_cur else ""
        )

        date_str = str(inv.invoice_date)[:10] if getattr(inv, "invoice_date", None) else ""

        cards.append(f"""
<div style="background:{row_bg}; border:1px solid {border}; border-radius:10px;
     padding:14px 16px; margin-bottom:8px; transition:border-color .2s;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div>
      <span style="color:#60a5fa; font-weight:600; font-family:monospace;
           font-size:0.88rem;">{inv_id}</span>{cur_tag}
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
      {_badge(status, sc, bg)}
      <span style="color:#334155; font-size:0.72rem;">{date_str}</span>
    </div>
  </div>
  <div style="font-size:1.05rem; font-weight:700; color:#f1f5f9; margin-top:5px;">
    ₹{inv.taxable_value:,.2f}
  </div>
  <div style="font-size:0.75rem; color:#475569; margin-top:2px;">{tax_str}</div>
  <div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px; margin-top:6px;">
    <span style="font-size:0.72rem; color:#334155; font-family:monospace;">
      {inv.supplier_gstin or "—"}</span>
    {flag_html}
  </div>
  {action_html}
</div>""")

    count  = len(_S.invoice_map)
    done_c = sum(1 for e in _S.invoice_map.values() if e.get("status", "pending") != "pending")
    pct    = int(done_c / count * 100) if count else 0

    header = f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
  <span style="font-size:0.65rem; color:#475569; text-transform:uppercase;
       letter-spacing:0.1em; font-weight:700;">INVOICES ({done_c}/{count} processed)</span>
  <div style="width:120px;">
    <div style="background:#0f172a; border-radius:3px; height:4px; overflow:hidden;">
      <div style="background:#3b82f6; height:100%; width:{pct}%; border-radius:3px;"></div>
    </div>
  </div>
</div>"""

    return header + "\n".join(cards)


def render_mismatches(obs: GSTObservation) -> str:
    if obs.task_id != 2:
        return (
            '<div style="color:#475569; text-align:center; padding:50px 20px;">'
            '<div style="font-size:2rem; margin-bottom:8px;">🔗</div>'
            '<div style="font-size:0.85rem;">Select <b>Task 2 — ITC Reconciliation</b> to see mismatches</div>'
            '</div>'
        )
    if not obs.mismatches:
        return (
            '<div style="color:#22c55e; text-align:center; padding:50px 20px;">'
            '<div style="font-size:2rem; margin-bottom:8px;">✓</div>'
            '<div style="font-size:0.9rem; font-weight:600;">All mismatches resolved</div>'
            '</div>'
        )

    cards = []
    for m in obs.mismatches:
        col = MISMATCH_TYPE_COLOR.get(m.mismatch_type, "#64748b")
        gstr2b_id = m.gstr2b_invoice_id or "NOT IN 2B"
        g2b_col   = "#ef4444" if gstr2b_id == "NOT IN 2B" else "#94a3b8"
        delta_col = "#ef4444" if abs(m.delta) > 5000 else "#f59e0b"

        cards.append(f"""
<div style="background:#131c2e; border:1px solid #1e293b; border-left:3px solid {col};
     border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:8px;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="color:#60a5fa; font-family:monospace; font-size:0.85rem;
         font-weight:600;">{m.purchase_invoice_id}</span>
    {_badge(m.mismatch_type.replace("_", " "), col, col + "1a")}
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; margin-top:8px;">
    <div>
      <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.05em;">GSTR-2B ID</div>
      <div style="font-size:0.8rem; color:{g2b_col}; font-weight:500; font-family:monospace;">{gstr2b_id}</div>
    </div>
    <div>
      <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.05em;">Purchase</div>
      <div style="font-size:0.8rem; color:#e2e8f0; font-weight:600;">₹{m.purchase_taxable:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.05em;">GSTR-2B</div>
      <div style="font-size:0.8rem; color:#e2e8f0; font-weight:600;">
        {f"₹{m.gstr2b_taxable:,.0f}" if m.gstr2b_taxable else "—"}</div>
    </div>
    <div>
      <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.05em;">Delta</div>
      <div style="font-size:0.8rem; color:{delta_col}; font-weight:700;">₹{m.delta:,.0f}</div>
    </div>
  </div>
</div>""")

    header = (
        f'<div style="font-size:0.65rem; color:#475569; text-transform:uppercase;'
        f' letter-spacing:0.1em; font-weight:700; margin-bottom:10px;">'
        f'ITC MISMATCHES ({len(obs.mismatches)} remaining)</div>'
    )
    return header + "\n".join(cards)


def render_filing(obs: GSTObservation) -> str:
    gstr     = obs.gstr3b
    done_set = set(gstr.sections_completed)
    completed = len(done_set)
    pct       = int(completed / 9 * 100)

    values = {
        "3.1a": gstr.taxable_outward, "3.1b": gstr.zero_rated,
        "3.1c": gstr.exempted,        "3.1d": gstr.rcm_inward,
        "4a":   gstr.itc_igst,        "4b":   gstr.itc_ineligible,
        "6.1":  gstr.igst_payable,    "6.2":  gstr.cgst_payable,
        "6.3":  gstr.sgst_payable,
    }

    rows = []
    for sec, desc in SECTION_INFO.items():
        is_done  = sec in done_set
        val      = values.get(sec, 0.0)
        s_col, s_bg = ("#22c55e", "#22c55e1a") if is_done else ("#f59e0b", "#f59e0b1a")
        row_bg   = "#0f1e14" if is_done else "#131c2e"
        val_col  = "#e2e8f0" if is_done else "#64748b"
        status_html = (
            f'<span style="color:{s_col}; font-size:0.75rem; font-weight:600;">'
            f'{"✓ Done" if is_done else "Pending"}</span>'
        )

        rows.append(f"""
<div style="background:{row_bg}; border:1px solid #1e293b; border-radius:8px;
     padding:10px 14px; margin-bottom:6px; display:flex; align-items:center; gap:10px;">
  <code style="width:44px; font-size:0.78rem; color:#60a5fa; flex-shrink:0;">{sec}</code>
  <span style="flex:1; font-size:0.78rem; color:#94a3b8;">{desc}</span>
  <span style="width:110px; text-align:right; font-size:0.82rem;
       font-weight:600; color:{val_col}; font-family:monospace;">₹{val:,.2f}</span>
  <span style="width:68px; text-align:right;">{status_html}</span>
</div>""")

    header = f"""
<div style="margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <span style="font-size:0.65rem; color:#475569; text-transform:uppercase;
         letter-spacing:0.1em; font-weight:700;">GSTR-3B SECTIONS ({completed}/9)</span>
    <span style="font-size:0.75rem; color:#94a3b8;">{pct}% complete</span>
  </div>
  <div style="background:#0f172a; border-radius:4px; height:5px; overflow:hidden;">
    <div style="background:#22c55e; height:100%; width:{pct}%;
         border-radius:4px; transition:width .5s ease;"></div>
  </div>
</div>"""

    return header + "\n".join(rows)


def render_log(entries: list[str]) -> str:
    if not entries:
        return (
            '<div style="color:#475569; text-align:center; padding:50px;">'
            '<div style="font-size:2rem; margin-bottom:8px;">📋</div>'
            '<div style="font-size:0.85rem;">Step log will appear here during the episode</div>'
            '</div>'
        )
    # newest first
    content = "\n".join(reversed(entries[-60:]))
    return f'<div style="font-family:monospace; font-size:0.78rem; line-height:1.6;">{content}</div>'


def render_summary(obs: GSTObservation, rewards: list, steps: int, done: bool) -> str:
    if not rewards and steps == 0:
        return (
            '<div style="color:#475569; text-align:center; padding:50px;">'
            '<div style="font-size:2rem; margin-bottom:8px;">📊</div>'
            '<div style="font-size:0.85rem;">Run an episode to see the summary</div>'
            '</div>'
        )

    total  = sum(rewards)
    pos    = sum(1 for r in rewards if r > 0)
    neg    = sum(1 for r in rewards if r < 0)
    avg    = total / len(rewards) if rewards else 0.0
    gstr   = obs.gstr3b

    s_col  = "#22c55e" if done else "#f59e0b"
    r_col  = "#22c55e" if total >= 0 else "#ef4444"

    def _row(label, value, color="#cbd5e1"):
        return (
            f'<div style="display:flex; justify-content:space-between; padding:8px 0;'
            f' border-bottom:1px solid #1e293b;">'
            f'<span style="color:#475569; font-size:0.8rem;">{label}</span>'
            f'<span style="color:{color}; font-size:0.8rem; font-weight:600;">{value}</span>'
            f'</div>'
        )

    rows = (
        _row("Status",        "✓ Complete" if done else f"⏳ Running ({steps} steps)", s_col) +
        _row("Total Reward",  f"{total:+.2f}", r_col) +
        _row("Avg / Step",    f"{avg:+.3f}") +
        _row("Steps Taken",   str(steps)) +
        _row("Positive Steps",f"{pos}", "#22c55e") +
        _row("Penalty Steps", f"{neg}", "#ef4444" if neg else "#64748b") +
        _row("Classified",    str(obs.classified_count)) +
        _row("Flagged",       str(obs.flagged_count)) +
        _row("Matched ITC",   f"₹{obs.matched_itc_amount:,.2f}") +
        _row("GSTR-3B Sections", f"{len(gstr.sections_completed)}/9")
    )

    return (
        f'<div style="background:#131c2e; border:1px solid #1e293b; border-radius:10px; padding:16px;">'
        f'{rows}</div>'
    )


def _make_log_entry(step: int, action, reward: float, info: dict) -> str:
    atype  = action.action_type.value
    icon   = ACTION_ICONS.get(atype, "▶")
    err    = info.get("last_action_error")
    r_col  = "#22c55e" if reward >= 0 else "#ef4444"
    err_html = (
        f'<div style="color:#ef4444; font-size:0.68rem; margin-top:2px;">⚠ {err}</div>'
        if err else ""
    )
    return (
        f'<div style="padding:7px 12px; border-bottom:1px solid #0f172a;'
        f' display:flex; align-items:flex-start; gap:8px;">'
        f'<span style="color:#334155; font-size:0.68rem; min-width:28px; padding-top:2px;">'
        f'{step:03d}</span>'
        f'<span style="min-width:18px; font-size:0.8rem;">{icon}</span>'
        f'<div style="flex:1;">'
        f'<div style="color:#64748b; font-size:0.75rem; word-break:break-all;">'
        f'{action.to_action_str()}</div>'
        f'{err_html}'
        f'</div>'
        f'<span style="color:{r_col}; font-weight:700; font-size:0.78rem;'
        f' min-width:48px; text-align:right;">{reward:+.2f}</span>'
        f'</div>'
    )


def _rewards_df(rewards: list[float]) -> pd.DataFrame:
    if not rewards:
        return pd.DataFrame(columns=["Step", "Reward", "Series"])
    rows = []
    cum = 0.0
    for i, r in enumerate(rewards):
        cum += r
        rows.append({"Step": i + 1, "Reward": r,   "Series": "Per-step"})
        rows.append({"Step": i + 1, "Reward": cum,  "Series": "Cumulative"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Episode logic
# ─────────────────────────────────────────────────────────────────────────────

def _task_id_from_choice(choice: str) -> int:
    return int(choice.split()[1])


def _all_outputs(obs: GSTObservation):
    """Pack all 9 output values from current state."""
    return (
        render_banner(_S.task_id, _S.step, _S.done),
        render_stats(_S.step, _S.max_steps, _S.rewards, obs.pending_count),
        render_context(obs),
        render_invoices(obs),
        render_mismatches(obs),
        render_filing(obs),
        render_log(_S.log_entries),
        _rewards_df(_S.rewards),
        render_summary(obs, _S.rewards, _S.step, _S.done),
    )


def on_reset(task_choice: str, seed: float):
    """Reset episode and return initial UI state."""
    task_id = _task_id_from_choice(task_choice)
    seed    = int(seed)

    if _S.env:
        try:
            _S.env.close()
        except Exception:
            pass

    _S.env       = GSTEnvironment()
    _S.obs, _    = _S.env.reset(task_id=task_id, seed=seed)
    _S.task_id   = task_id
    _S.seed      = seed
    _S.step      = 0
    _S.max_steps = TASK_INFO[task_id]["max_steps"]
    _S.rewards   = []
    _S.log_entries = []
    _S.done      = False
    _S.invoice_map = {}

    # Pre-populate invoice map from env state (task 1 & 2)
    state = getattr(_S.env, "_state", {})
    if task_id == 1:
        for inv in state.get("invoices", []):
            _S.invoice_map[inv.invoice_id] = {"inv": inv, "status": "pending"}
    elif task_id == 2:
        seen = set()
        for m in state.get("mismatches", []):
            iid = m.purchase_invoice_id
            if iid not in seen:
                seen.add(iid)
                # create a minimal stub for display
                from models.observation import InvoiceObservation, InvoiceStatus
                stub = InvoiceObservation(
                    invoice_id=iid,
                    invoice_type="B2B",
                    taxable_value=m.purchase_taxable,
                    igst_amount=0.0,
                    cgst_amount=0.0,
                    sgst_amount=0.0,
                    total_amount=m.purchase_taxable,
                    supplier_gstin="",
                    buyer_gstin="",
                    invoice_date="—",
                    hsn_code="",
                    flags=[m.mismatch_type],
                    description="",
                    status=InvoiceStatus.PENDING,
                )
                _S.invoice_map[iid] = {"inv": stub, "status": "pending"}

    return _all_outputs(_S.obs)


def on_run(task_choice: str, seed: float) -> Generator:
    """Run a full episode with the baseline agent, streaming updates."""
    # ensure a fresh episode
    yield from _stream_episode(task_choice, seed)


def _stream_episode(task_choice: str, seed: float):
    task_id = _task_id_from_choice(task_choice)
    seed    = int(seed)

    # reset first
    on_reset(task_choice, seed)
    yield _all_outputs(_S.obs)

    agent = BaselineAgent()

    while _S.step < _S.max_steps:
        action = agent.act(_S.obs, task_id)
        _S.obs, reward, terminated, truncated, info = _S.env.step(action)
        _S.rewards.append(reward)
        _S.step += 1

        # update invoice status in map
        if action.action_type.value in ("classify_invoice", "skip_invoice") and hasattr(action.payload, "invoice_id"):
            iid = action.payload.invoice_id
            if iid in _S.invoice_map:
                new_status = "classified" if action.action_type.value == "classify_invoice" else "skipped"
                _S.invoice_map[iid]["status"] = new_status
                _S.invoice_map[iid]["action"] = action.to_action_str()
                _S.invoice_map[iid]["reward"] = reward
        elif action.action_type.value == "flag_discrepancy" and hasattr(action.payload, "invoice_id"):
            iid = action.payload.invoice_id
            if iid in _S.invoice_map:
                _S.invoice_map[iid]["status"] = "flagged"
                _S.invoice_map[iid]["action"] = action.to_action_str()
                _S.invoice_map[iid]["reward"] = reward
        elif action.action_type.value == "match_itc" and hasattr(action.payload, "purchase_invoice_id"):
            iid = action.payload.purchase_invoice_id
            if iid in _S.invoice_map:
                _S.invoice_map[iid]["status"] = "matched"
                _S.invoice_map[iid]["action"] = action.to_action_str()
                _S.invoice_map[iid]["reward"] = reward
        elif action.action_type.value == "accept_mismatch" and hasattr(action.payload, "purchase_invoice_id"):
            iid = action.payload.purchase_invoice_id
            if iid in _S.invoice_map:
                _S.invoice_map[iid]["status"] = "accepted"
                _S.invoice_map[iid]["action"] = action.to_action_str()
                _S.invoice_map[iid]["reward"] = reward
        elif action.action_type.value == "defer_invoice" and hasattr(action.payload, "invoice_id"):
            iid = action.payload.invoice_id
            if iid in _S.invoice_map:
                _S.invoice_map[iid]["status"] = "deferred"
                _S.invoice_map[iid]["action"] = action.to_action_str()
                _S.invoice_map[iid]["reward"] = reward

        _S.log_entries.append(_make_log_entry(_S.step, action, reward, info))

        done = terminated or truncated
        _S.done = done

        yield _all_outputs(_S.obs)

        if done:
            break

    _S.env.close()
    _S.done = True
    yield _all_outputs(_S.obs)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ── */
body, .gradio-container { background:#0a0f1a !important; color:#e2e8f0 !important; }
footer { display:none !important; }
.tabs > .tab-nav { background:#0d1424 !important; border-bottom:1px solid #1e293b !important; }
.tabs > .tab-nav button { color:#64748b !important; font-size:0.82rem !important;
    font-weight:500 !important; padding:8px 16px !important; }
.tabs > .tab-nav button.selected { color:#e2e8f0 !important;
    border-bottom:2px solid #3b82f6 !important; background:transparent !important; }
/* ── Sidebar ── */
#sidebar { background:#0d1424 !important; border-right:1px solid #1e293b !important;
    padding:0 !important; }
#sidebar .block { background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 12px !important; }
#sidebar label { color:#64748b !important; font-size:0.72rem !important;
    font-weight:600 !important; text-transform:uppercase !important;
    letter-spacing:0.08em !important; }
/* ── Section labels ── */
.sec-label { font-size:0.62rem !important; font-weight:700 !important; letter-spacing:0.12em !important;
    text-transform:uppercase !important; color:#334155 !important; padding:14px 12px 6px !important;
    border-top:1px solid #1e293b !important; margin-top:4px !important; }
.sec-label:first-child { border-top:none !important; margin-top:0 !important; }
/* ── Inputs ── */
select, input[type=number], .gr-dropdown select {
    background:#0f172a !important; border:1px solid #1e293b !important;
    color:#e2e8f0 !important; border-radius:8px !important; font-size:0.82rem !important; }
/* ── Buttons ── */
.reset-btn { background:#1d4ed8 !important; border:none !important;
    border-radius:8px !important; color:white !important; font-weight:600 !important;
    font-size:0.82rem !important; padding:9px !important; }
.run-btn { background:#064e3b !important; border:1px solid #065f46 !important;
    border-radius:8px !important; color:#34d399 !important; font-weight:600 !important;
    font-size:0.82rem !important; padding:9px !important; }
/* ── Main panel ── */
#main-panel { background:#0a0f1a !important; padding:12px 16px !important; }
#main-panel .block { background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important; }
/* ── Tab content scrollable ── */
.tab-content { max-height:calc(100vh - 220px); overflow-y:auto; padding:12px 4px !important; }
/* ── Reward chart ── */
.gr-plot { background:#131c2e !important; border:1px solid #1e293b !important;
    border-radius:10px !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="GST Sahayak — RL Environment",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=CSS,
    ) as demo:

        # ── App header ────────────────────────────────────────────────────────
        gr.HTML("""
<div style="background:#0d1424; border-bottom:1px solid #1e293b;
     padding:14px 20px; display:flex; align-items:center; justify-content:space-between;">
  <div>
    <span style="font-size:1.15rem; font-weight:700; color:#f1f5f9;">GST Sahayak</span>
    <span style="font-size:0.75rem; color:#475569; margin-left:10px;">
      RL Environment — Meta OpenEnv Hackathon</span>
  </div>
  <div style="display:flex; gap:8px;">
    <span style="background:#3b82f620; color:#60a5fa; border:1px solid #3b82f640;
         padding:3px 10px; border-radius:20px; font-size:0.65rem; font-weight:600;">openenv</span>
    <span style="background:#22c55e20; color:#4ade80; border:1px solid #22c55e40;
         padding:3px 10px; border-radius:20px; font-size:0.65rem; font-weight:600;">API: online</span>
  </div>
</div>""")

        with gr.Row(equal_height=False):

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=230, elem_id="sidebar"):

                gr.HTML('<div class="sec-label" style="border-top:none;">Episode Setup</div>')

                task_dd = gr.Dropdown(
                    choices=[
                        "Task 1 — Invoice Classifier (Easy)",
                        "Task 2 — ITC Reconciliation (Medium)",
                        "Task 3 — GSTR-3B Filing (Hard)",
                    ],
                    value="Task 1 — Invoice Classifier (Easy)",
                    label="Task",
                    container=True,
                )
                seed_num = gr.Number(
                    value=42, minimum=0, maximum=9999, precision=0,
                    label="Seed",
                )
                reset_btn = gr.Button(
                    "↺  Reset / New Episode",
                    elem_classes=["reset-btn"],
                    size="sm",
                )
                run_btn = gr.Button(
                    "▶  Auto Run (Agent)",
                    elem_classes=["run-btn"],
                    size="sm",
                )

                gr.HTML('<div class="sec-label">Episode Stats</div>')
                stats_out = gr.HTML(
                    render_stats(0, 50, [], 0)
                )

                gr.HTML('<div class="sec-label">Context</div>')
                ctx_out = gr.HTML(
                    '<div style="color:#334155; font-size:0.75rem; padding:4px 0;">'
                    'Reset to load context.</div>'
                )

                # Task description accordion
                gr.HTML('<div class="sec-label">Task Info</div>')
                task_info_html = gr.HTML("""
<div style="padding:4px 0;">
  <div style="background:#162032; border:1px solid #1e293b; border-radius:8px; padding:10px 12px;">
    <div style="font-size:0.75rem; color:#94a3b8; line-height:1.5;">
      Classify each invoice: type (B2B/B2C/RCM…), HSN code, GST slab, ITC eligibility,
      and reverse-charge flag.
    </div>
    <div style="font-size:0.68rem; color:#475569; margin-top:8px; font-style:italic;">
      +0.40 correct slab · +0.25 correct HSN · −0.30 wrong slab
    </div>
  </div>
</div>""")

            # ── RIGHT MAIN PANEL ─────────────────────────────────────────────
            with gr.Column(scale=3, elem_id="main-panel"):

                banner_out = gr.HTML(
                    render_banner(1, 0, False)
                )

                with gr.Tabs(elem_id="main-tabs"):

                    with gr.Tab("Invoices"):
                        with gr.Column(elem_classes=["tab-content"]):
                            invoices_out = gr.HTML(
                                '<div style="color:#475569; text-align:center; padding:50px;">'
                                '<div style="font-size:2rem;">📄</div>'
                                '<div style="margin-top:8px; font-size:0.85rem;">Press Reset to load episode</div>'
                                '</div>'
                            )

                    with gr.Tab("GSTR-2B"):
                        with gr.Column(elem_classes=["tab-content"]):
                            mismatch_out = gr.HTML(
                                '<div style="color:#475569; text-align:center; padding:50px;">'
                                '<div style="font-size:2rem;">🔗</div>'
                                '<div style="margin-top:8px; font-size:0.85rem;">Select Task 2 for ITC mismatches</div>'
                                '</div>'
                            )

                    with gr.Tab("GSTR-3B"):
                        with gr.Column(elem_classes=["tab-content"]):
                            filing_out = gr.HTML("")

                    with gr.Tab("Step Log"):
                        with gr.Column(elem_classes=["tab-content"]):
                            log_out = gr.HTML(
                                '<div style="color:#475569; text-align:center; padding:50px;">'
                                '<div style="font-size:2rem;">📋</div>'
                                '<div style="margin-top:8px; font-size:0.85rem;">Episode log appears here</div>'
                                '</div>'
                            )

                    with gr.Tab("Rewards"):
                        with gr.Column(elem_classes=["tab-content"]):
                            reward_plot = gr.LinePlot(
                                x="Step",
                                y="Reward",
                                color="Series",
                                title="Rewards per Step",
                                x_lim=[0, None],
                                tooltip=["Step", "Reward", "Series"],
                            )

                    with gr.Tab("Summary"):
                        with gr.Column(elem_classes=["tab-content"]):
                            summary_out = gr.HTML(
                                '<div style="color:#475569; text-align:center; padding:50px;">'
                                '<div style="font-size:2rem;">📊</div>'
                                '<div style="margin-top:8px; font-size:0.85rem;">Run an episode to see results</div>'
                                '</div>'
                            )

        # ── Task info updater ─────────────────────────────────────────────────
        def update_task_info(task_choice):
            tid  = _task_id_from_choice(task_choice)
            info = TASK_INFO[tid]
            col  = info["difficulty_color"]
            return (
                f'<div style="padding:4px 0;">'
                f'<div style="background:#162032; border:1px solid #1e293b; border-left:3px solid {col};'
                f' border-radius:0 8px 8px 0; padding:10px 12px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                f'<span style="font-size:0.75rem; font-weight:600; color:#e2e8f0;">{info["full_name"]}</span>'
                f'<span style="background:{col}20; color:{col}; border:1px solid {col}40;'
                f' padding:1px 7px; border-radius:20px; font-size:0.62rem; font-weight:700;">{info["difficulty"]}</span>'
                f'</div>'
                f'<div style="font-size:0.73rem; color:#94a3b8; line-height:1.5;">{info["description"]}</div>'
                f'<div style="font-size:0.67rem; color:#475569; margin-top:8px; font-style:italic;">{info["reward_tips"]}</div>'
                f'</div></div>'
            )

        task_dd.change(fn=update_task_info, inputs=[task_dd], outputs=[task_info_html])

        # ── Output list (shared by reset + run) ───────────────────────────────
        _OUTPUTS = [
            banner_out, stats_out, ctx_out,
            invoices_out, mismatch_out, filing_out,
            log_out, reward_plot, summary_out,
        ]

        reset_btn.click(fn=on_reset, inputs=[task_dd, seed_num], outputs=_OUTPUTS)
        run_btn.click(fn=on_run,    inputs=[task_dd, seed_num], outputs=_OUTPUTS)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Mount on FastAPI (keeps /reset /step /state /close for openenv)
# ─────────────────────────────────────────────────────────────────────────────

from server import app as fastapi_app

demo = build_ui()
app  = gr.mount_gradio_app(fastapi_app, demo, path="/")


# ─────────────────────────────────────────────────────────────────────────────
# Local dev
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
