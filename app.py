"""
app.py — Gradio UI for GST Intelligence RL

Mounts on top of the FastAPI server so /reset, /step, /state, /close
endpoints are still available for openenv validate.

Entry point for HuggingFace Space (replaces server.py in Dockerfile).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Generator

import gradio as gr

from env.gst_env import GSTEnvironment
from models.observation import GSTObservation
from agent.baseline_agent import BaselineAgent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_INFO = {
    1: {
        "name": "Invoice Classification",
        "difficulty": "Easy",
        "max_steps": 50,
        "color": "#22c55e",
        "description": (
            "The agent reads each invoice and decides: "
            "invoice type (B2B/B2C/RCM…), HSN code, GST slab (0/5/12/18/28%), "
            "ITC eligibility, and reverse charge flag."
        ),
        "reward_tips": "+0.40 correct slab · +0.25 correct HSN · −0.30 wrong slab",
    },
    2: {
        "name": "ITC Reconciliation",
        "difficulty": "Medium",
        "max_steps": 150,
        "color": "#f59e0b",
        "description": (
            "The agent matches purchase register invoices against GSTR-2B. "
            "It must detect amount mismatches, fake invoices, cancelled entries, "
            "and invoices missing from GSTR-2B."
        ),
        "reward_tips": "+0.50 correct match · −0.40 false match · −0.30 missed fraud",
    },
    3: {
        "name": "GSTR-3B Filing",
        "difficulty": "Hard",
        "max_steps": 300,
        "color": "#ef4444",
        "description": (
            "The agent fills all 9 GSTR-3B sections in legal order, "
            "applies ITC offset rules (IGST→IGST/CGST/SGST), "
            "and submits the return within the step budget."
        ),
        "reward_tips": "+0.20 section within 5% · −1.00 ITC overclaim (episode ends)",
    },
}

ACTION_ICONS = {
    "classify_invoice":      "🏷️",
    "match_itc":             "🔗",
    "flag_discrepancy":      "🚩",
    "accept_mismatch":       "✅",
    "defer_invoice":         "⏳",
    "set_section_value":     "📝",
    "generate_return":       "⚙️",
    "submit_return":         "📤",
    "skip_invoice":          "⏭️",
    "request_clarification": "❓",
}

SECTION_LABELS = {
    "3.1a": "3.1(a) Outward taxable",
    "3.1b": "3.1(b) Zero-rated",
    "3.1c": "3.1(c) Exempt",
    "3.1d": "3.1(d) RCM inward",
    "4a":   "4(A) ITC available",
    "4b":   "4(B) ITC reversed",
    "6.1":  "6.1 IGST payable",
    "6.2":  "6.2 CGST payable",
    "6.3":  "6.3 SGST payable",
}


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner (generator — yields UI updates after every step)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_id: int, seed: int) -> Generator:
    """
    Runs one full episode with the baseline agent.
    Yields (log_md, invoice_table, mismatch_table, filing_table,
            reward_chart_data, summary_md) after every step.
    """
    env   = GSTEnvironment()
    agent = BaselineAgent()

    obs, _ = env.reset(task_id=task_id, seed=seed)
    max_steps = TASK_INFO[task_id]["max_steps"]

    log_lines: list[str] = []
    rewards:   list[float] = []
    step = 0

    info_task = TASK_INFO[task_id]
    log_lines.append(
        f"### Episode started\n"
        f"**Task {task_id} — {info_task['name']}** · seed={seed} · max_steps={max_steps}\n"
        f"GSTIN: `{obs.gstin}` · Period: `{obs.tax_period}`\n\n"
        f"---"
    )
    yield _pack(log_lines, obs, rewards, task_id, done=False, summary="")

    while step < max_steps:
        action = agent.act(obs, task_id)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1

        atype     = action.action_type.value
        icon      = ACTION_ICONS.get(atype, "▶️")
        act_str   = action.to_action_str()
        err       = info.get("last_action_error")
        err_str   = f" ⚠️ `{err}`" if err else ""
        sub_r     = info.get("sub_rewards", {})
        sub_str   = "  ".join(f"`{k}` {v:+.2f}" for k, v in sub_r.items()) if sub_r else ""

        reward_color = "green" if reward >= 0 else "red"
        log_lines.append(
            f"**Step {step}** {icon} `{act_str}`{err_str}\n"
            f"reward: <span style='color:{reward_color}'>{reward:+.2f}</span> · "
            f"cumulative: **{sum(rewards):.2f}**"
            + (f"\n> {sub_str}" if sub_str else "")
        )

        done = terminated or truncated
        summary = _build_summary(obs, rewards, step, done, task_id)
        yield _pack(log_lines, obs, rewards, task_id, done=done, summary=summary)

        if done:
            break

    env.close()


def _pack(log_lines, obs, rewards, task_id, done, summary):
    log_md       = "\n\n".join(log_lines[-30:])   # keep last 30 steps visible
    inv_rows     = _invoice_rows(obs)
    mismatch_rows = _mismatch_rows(obs)
    filing_rows  = _filing_rows(obs)
    reward_data  = [[i + 1, r, sum(rewards[:i+1])] for i, r in enumerate(rewards)]
    return log_md, inv_rows, mismatch_rows, filing_rows, reward_data, summary


def _build_summary(obs, rewards, steps, done, task_id):
    total = sum(rewards)
    pos   = sum(1 for r in rewards if r > 0)
    neg   = sum(1 for r in rewards if r < 0)
    status = "**Episode complete**" if done else f"Running… step {steps}"
    return (
        f"{status}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Steps taken | {steps} |\n"
        f"| Cumulative reward | **{total:.2f}** |\n"
        f"| Positive steps | {pos} |\n"
        f"| Penalty steps | {neg} |\n"
        f"| Classified | {obs.classified_count} |\n"
        f"| Flagged | {obs.flagged_count} |\n"
        f"| Matched ITC | ₹{obs.matched_itc_amount:,.2f} |\n"
        f"| Sections done | {len(obs.gstr3b.sections_completed)} / 9 |"
    )


def _invoice_rows(obs):
    rows = []
    invoices = []
    # current invoice always first
    if obs.current_invoice:
        invoices = [obs.current_invoice]
    for inv in invoices:
        rows.append([
            inv.invoice_id,
            inv.invoice_type,
            inv.hsn_code or "—",
            f"₹{inv.taxable_value:,.0f}",
            f"₹{inv.igst_amount:,.0f}" if inv.igst_amount else "—",
            inv.status.value,
            ", ".join(inv.flags) if inv.flags else "—",
        ])
    return rows


def _mismatch_rows(obs):
    rows = []
    for m in obs.mismatches[:10]:
        rows.append([
            m.purchase_invoice_id,
            m.gstr2b_invoice_id or "NOT IN 2B",
            m.mismatch_type,
            f"₹{m.purchase_taxable:,.0f}",
            f"₹{m.gstr2b_taxable:,.0f}" if m.gstr2b_taxable else "—",
            f"₹{m.delta:,.0f}",
        ])
    return rows


def _filing_rows(obs):
    gstr = obs.gstr3b
    mapping = {
        "3.1a": gstr.taxable_outward,
        "3.1b": gstr.zero_rated,
        "3.1c": gstr.exempted,
        "3.1d": gstr.rcm_inward,
        "4a":   gstr.itc_igst,
        "4b":   gstr.itc_ineligible,
        "6.1":  gstr.igst_payable,
        "6.2":  gstr.cgst_payable,
        "6.3":  gstr.sgst_payable,
    }
    rows = []
    done = set(gstr.sections_completed)
    for sec, label in SECTION_LABELS.items():
        val    = mapping.get(sec, 0.0)
        status = "Done" if sec in done else "Pending"
        rows.append([sec, label, f"₹{val:,.2f}", status])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    CSS = """
    .task-card { border-radius: 12px; padding: 16px; margin: 4px 0; }
    .header-title { font-size: 2rem; font-weight: 700; }
    .step-log { font-size: 0.85rem; font-family: monospace; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="GST Intelligence RL") as demo:

        # ── Header ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding: 24px 0 8px 0;">
          <h1 style="font-size:2rem; font-weight:700; margin:0;">
            📊 GST Intelligence RL
          </h1>
          <p style="color:#64748b; margin-top:6px;">
            An RL agent that automates Indian GST workflows —
            invoice classification, ITC reconciliation &amp; GSTR-3B filing
          </p>
        </div>
        """)

        # ── Task cards ───────────────────────────────────────────────────────
        with gr.Row():
            for tid, info in TASK_INFO.items():
                gr.HTML(f"""
                <div class="task-card" style="background:#f8fafc; border-left: 4px solid {info['color']};">
                  <div style="font-weight:600; font-size:1rem;">
                    Task {tid} · {info['name']}
                    <span style="float:right; background:{info['color']}22;
                                 color:{info['color']}; border-radius:20px;
                                 padding:2px 10px; font-size:0.75rem;">
                      {info['difficulty']}
                    </span>
                  </div>
                  <div style="color:#475569; font-size:0.85rem; margin-top:6px;">
                    {info['description']}
                  </div>
                  <div style="color:#94a3b8; font-size:0.75rem; margin-top:8px;">
                    {info['reward_tips']}
                  </div>
                </div>
                """)

        gr.HTML("<hr style='margin: 8px 0; border-color:#e2e8f0;'>")

        # ── Controls ─────────────────────────────────────────────────────────
        with gr.Row():
            task_radio = gr.Radio(
                choices=["Task 1 — Invoice Classification (Easy)",
                         "Task 2 — ITC Reconciliation (Medium)",
                         "Task 3 — GSTR-3B Filing (Hard)"],
                value="Task 1 — Invoice Classification (Easy)",
                label="Select Task",
                interactive=True,
            )
            seed_slider = gr.Slider(
                minimum=0, maximum=999, value=42, step=1,
                label="Episode Seed (controls which invoices are generated)",
                interactive=True,
            )
            run_btn = gr.Button("▶  Run Episode", variant="primary", scale=0)

        # ── Main content tabs ─────────────────────────────────────────────────
        with gr.Tabs():

            # Tab 1 — Step-by-step log
            with gr.Tab("Step Log"):
                log_box = gr.Markdown(
                    value="*Press **Run Episode** to start.*",
                    label="Agent Steps",
                    elem_classes=["step-log"],
                )

            # Tab 2 — Current invoice
            with gr.Tab("Current Invoice"):
                gr.Markdown("The invoice the agent is currently processing.")
                inv_table = gr.Dataframe(
                    headers=["Invoice ID", "Type", "HSN", "Taxable", "IGST", "Status", "Flags"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

            # Tab 2b — Mismatches (Task 2)
            with gr.Tab("ITC Mismatches"):
                gr.Markdown("Purchase register vs GSTR-2B mismatches remaining.")
                mismatch_table = gr.Dataframe(
                    headers=["Purchase ID", "GSTR-2B ID", "Mismatch Type",
                              "Purchase Value", "GSTR-2B Value", "Delta"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

            # Tab 3 — GSTR-3B sections
            with gr.Tab("GSTR-3B Sections"):
                gr.Markdown("Filing progress — sections the agent has filled.")
                filing_table = gr.Dataframe(
                    headers=["Section", "Description", "Value", "Status"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

            # Tab 4 — Reward chart
            with gr.Tab("Reward Chart"):
                gr.Markdown("Per-step reward and cumulative reward over the episode.")
                reward_plot = gr.LinePlot(
                    x="Step",
                    y="Reward",
                    color="Series",
                    title="Rewards per Step",
                    x_lim=[0, None],
                    tooltip=["Step", "Reward", "Series"],
                )

            # Tab 5 — Summary
            with gr.Tab("Summary"):
                summary_box = gr.Markdown("*Run an episode to see results.*")

        # ── How it works ─────────────────────────────────────────────────────
        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
**Observation** → The agent receives a `GSTObservation` Pydantic object with the current invoice,
pending mismatches, GSTR-3B state, memory of past decisions, and supplier profile.

**Action** → The agent picks one typed action (`GSTAction`) — classify, match, flag, defer, set section, or submit.

**Reward** → A `RewardEngine` computes dense per-step rewards with potential-based shaping.
No sparse rewards — every step gives signal.

**Tasks**
- **Task 1** (Easy): 20 invoices, max 50 steps — classify each with HSN + slab + type
- **Task 2** (Medium): ~30 invoices, max 150 steps — reconcile purchase register vs GSTR-2B
- **Task 3** (Hard): 1 aggregate return, max 300 steps — fill 9 sections in legal order and submit

**ITC Offset Rules** (legally mandated in India)
```
IGST ITC  →  offsets IGST first, then CGST, then SGST
CGST ITC  →  offsets CGST only
SGST ITC  →  offsets SGST only
```
            """)

        # ── Wiring ───────────────────────────────────────────────────────────

        def on_run(task_choice: str, seed: int):
            tid = int(task_choice.split()[1])
            for log_md, inv_rows, mismatch_rows, filing_rows, rdata, summary in run_episode(tid, int(seed)):
                # Build reward dataframe for LinePlot
                import pandas as pd
                rows = []
                for step, per_r, cum_r in rdata:
                    rows.append({"Step": step, "Reward": per_r,  "Series": "Per-step"})
                    rows.append({"Step": step, "Reward": cum_r,  "Series": "Cumulative"})
                df = pd.DataFrame(rows) if rows else pd.DataFrame(
                    columns=["Step", "Reward", "Series"])

                yield log_md, inv_rows, mismatch_rows, filing_rows, df, summary

        run_btn.click(
            fn=on_run,
            inputs=[task_radio, seed_slider],
            outputs=[log_box, inv_table, mismatch_table, filing_table, reward_plot, summary_box],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Mount Gradio on FastAPI (keeps /reset /step /state /close for openenv)
# ─────────────────────────────────────────────────────────────────────────────

from server import app as fastapi_app  # FastAPI instance

demo = build_ui()
app  = gr.mount_gradio_app(fastapi_app, demo, path="/")


# ─────────────────────────────────────────────────────────────────────────────
# Local dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
