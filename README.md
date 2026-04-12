---
title: GST Intelligence RL
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - gst
  - tax-automation
  - india
license: mit
---

# GST Intelligence RL Environment

An **OpenEnv-compliant** Reinforcement Learning environment for automated Indian GST workflows.

The agent learns to perform three real-world GST compliance tasks — from invoice classification to full GSTR-3B filing — with dense rewards, deterministic graders, and a synthetic dataset of 1,000 seeded episodes.

---

## Tasks

| # | Name | Difficulty | Max Steps | Description |
|---|------|-----------|-----------|-------------|
| 1 | `invoice_classification` | Easy | 50 | Classify invoices: HSN code, GST slab, invoice type, ITC eligibility |
| 2 | `itc_reconciliation` | Medium | 150 | Match purchase register vs GSTR-2B, flag discrepancies |
| 3 | `gstr3b_filing` | Hard | 300 | Fill all GSTR-3B sections and submit the return |

---

## Environment API

```python
from env.gst_env import GSTEnvironment

env = GSTEnvironment()
obs, info = env.reset(task_id=1, seed=42)

# obs is a GSTObservation Pydantic model
print(obs.current_invoice)
print(obs.pending_count)

obs, reward, terminated, truncated, info = env.step(action)
state = env.state()
env.close()
```

### HTTP API (when deployed)

```
POST /reset   {"task_id": 1, "seed": 42}
POST /step    <GSTAction JSON>
GET  /state
POST /close
GET  /health
```

---

## Action Space

| Action | Phase | Description |
|--------|-------|-------------|
| `classify_invoice` | CLASSIFYING | Set invoice_type, hsn_code, gst_slab, itc_eligible, reverse_charge |
| `match_itc` | RECONCILING | Match purchase invoice to GSTR-2B entry |
| `flag_discrepancy` | RECONCILING | Flag mismatch with type + recommended action |
| `accept_mismatch` | RECONCILING | Accept immaterial difference (delta < ₹1,000) |
| `defer_invoice` | RECONCILING | Defer not-in-2B invoice awaiting supplier filing |
| `set_section_value` | FILING | Set a GSTR-3B section value |
| `generate_return` | FILING | Compute net payable with ITC offsets |
| `submit_return` | FILING | Submit the GSTR-3B return |
| `skip_invoice` | ANY | Skip current invoice (penalty: −0.10) |

---

## Reward Design

**Dense rewards** — every step gives feedback.

### Task 1 — Classification
```
+0.40  GST slab correct
+0.25  HSN code correct  (partial: 4-digit prefix = +0.125)
+0.15  Invoice type correct
+0.10  ITC eligibility correct
+0.10  Reverse charge correct
−0.30  Wrong slab
−0.20  Invalid HSN
−0.10  Skip used
+0.05  Efficiency bonus (steps remaining / max_steps)
```

### Task 2 — ITC Reconciliation
```
+0.50  Correct ITC match
+0.20  Amount delta < ₹1,000
+0.15  Correct discrepancy type
+0.15  Correct recommended action
−0.40  False match (different suppliers)
−0.30  Missed fraud flag
−0.20  Duplicate action
−0.10  Unnecessary defer
```

### Task 3 — GSTR-3B Filing
```
Per section:  +0.20 if error < 5%,  +0.10 if error < 20%,  −0.10 otherwise
On submit:    +0.50 all sections within 5%
              +0.25 ITC offset correct
              +0.15 net payable within 2%
              −0.50 unresolved discrepancies
              −0.30 underpayment
              −1.00 ITC overclaim (episode-ending)
```

---

## Data

| Split | Episodes | Seed | Task distribution |
|-------|----------|------|------------------|
| train | 800 | 42 | Round-robin 1/2/3 |
| val | 100 | 999 | Round-robin 1/2/3 |
| test | 100 | Curated | 34 / 33 / 33 |

Episodes are pre-generated and stored at `data/{split}/{task_id}/episode_NNNN.json`.

---

## Agents

### Baseline (Rule-Based)
Deterministic rule agent using HSN lookup table and hard-coded GST logic.
```bash
python -c "
from agent.baseline_agent import BaselineAgent
from env.gst_env import GSTEnvironment
env = GSTEnvironment()
agent = BaselineAgent()
obs, _ = env.reset(task_id=1, seed=42)
action = agent.act(obs, task_id=1)
print(action.to_action_str())
"
```

### RL Agent (PPO)
```bash
# Train
python agent/rl_agent.py --train --task 1 --timesteps 500000

# Evaluate
python agent/rl_agent.py --eval --task 1 --model checkpoints/ppo_gst_task1_final
```

---

## Running Inference

```bash
# Requires: HF_TOKEN, API_BASE_URL, MODEL_NAME environment variables
export HF_TOKEN=your_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini

python inference.py --task 1 --seed 42
```

Expected output format:
```
[START] task=invoice_classification env=gst-intelligence model=gpt-4.1-mini
[STEP] step=1 action=classify_invoice(INV-...,B2B,8471,18) reward=0.85 done=false error=null
...
[END] success=true steps=20 rewards=0.85,0.90,...
```

---

## GST Concepts Implemented

- **HSN/SAC codes** — 95-entry curated lookup table with slab mappings
- **GSTIN validation** — Mod-36 checksum (fake invoice detection)
- **ITC eligibility** — Section 17(5) ineligible items
- **RCM** — Reverse Charge Mechanism (recipient pays tax)
- **ITC offset rules** — IGST → IGST/CGST/SGST, CGST → CGST, SGST → SGST
- **GSTR-2B reconciliation** — Fuzzy matching (GSTIN + amount + date + invoice number)
- **GSTR-3B sections** — 3.1(a/b/c/d), 4(A)(5)/4(B), 6.1/6.2/6.3

---

## Project Structure

```
gst_rl/
├── inference.py          ← Hackathon entry point
├── openenv.yaml          ← OpenEnv validate target
├── server.py             ← FastAPI HTTP wrapper
├── Dockerfile
├── README.md
├── requirements.txt
├── env/
│   ├── gst_env.py        ← GSTEnvironment (Gymnasium)
│   ├── rewards.py        ← Dense reward engine + potential shaping
│   ├── memory_manager.py ← Per-episode invoice history
│   ├── tasks/            ← Task 1/2/3 apply_action + is_done
│   └── graders/          ← Deterministic graders (accuracy, precision/recall)
├── models/
│   ├── observation.py    ← GSTObservation (Pydantic v2)
│   ├── action.py         ← GSTAction (Pydantic v2)
│   └── reward.py         ← RewardSignal (Pydantic v2)
├── data/
│   ├── synthetic/        ← Seeded invoice generator + HSN table
│   ├── semantic/         ← GST rules JSON
│   ├── train/            ← 800 pre-generated episodes
│   ├── val/              ← 100 episodes (held-out)
│   └── test/             ← 100 curated episodes
└── agent/
    ├── baseline_agent.py ← Deterministic rule-based agent
    └── rl_agent.py       ← PPO agent (stable-baselines3)
```

---

## License

MIT
