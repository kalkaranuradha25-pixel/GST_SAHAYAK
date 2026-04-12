# GST Intelligence RL System — Product Requirements Document (PRD)

> Version: 1.1 | Hackathon Edition — Meta OpenEnv RL Challenge  
> Stack: Python 3.11, Gymnasium, Pydantic v2, FastAPI, Docker, HuggingFace Spaces  
> Compliance: OpenEnv Spec · Meta Hackathon Guidelines v1.0

---

## ⚠️ HACKATHON COMPLIANCE CHECKLIST (READ FIRST)

Every item below is a **submission gate** — missing any one causes automatic failure.

| Requirement | Where Implemented | Status |
|-------------|------------------|--------|
| `inference.py` in root directory | `/inference.py` | Required |
| OpenAI client only (no direct HTTP) | `inference.py` | Required |
| `API_BASE_URL` env var with default | `inference.py` | Required |
| `MODEL_NAME` env var with default | `inference.py` | Required |
| `HF_TOKEN` env var, no default | `inference.py` | Required |
| `[START]` line emitted at episode begin | `inference.py` | Required |
| `[STEP]` line per step after `env.step()` | `inference.py` | Required |
| `[END]` line after `env.close()`, always | `inference.py` | Required |
| `openenv validate` passes | `openenv.yaml` + `env/gst_env.py` | Required |
| Pydantic Observation, Action, Reward | `models/` | Required |
| `step(action) → (obs, reward, done, info)` | `env/gst_env.py` | Required |
| `reset() → initial observation` | `env/gst_env.py` | Required |
| `state() → current state dict` | `env/gst_env.py` | Required |
| 3 tasks, easy → medium → hard | `env/tasks/` | Required |
| Deterministic grader, score ∈ [0,1] | `env/graders/` | Required |
| Dense reward (not sparse) | `env/rewards.py` | Required |
| Working `Dockerfile` | `/Dockerfile` | Required |
| HF Space live and Running | HuggingFace | Required |
| 2 vCPU, 8GB RAM constraint | `Dockerfile` | Required |
| `README.md` with full docs | `/README.md` | Required |

---

## 1. Users

| User | Pain Point | How System Helps |
|------|-----------|-----------------|
| **SME Accountants** | Manual GSTR-2B vs purchase register matching: 3–10 days/month | Automated ITC reconciliation in < 5 minutes |
| **CA / Tax Consultants** | Invoice misclassification → penalty exposure | RL agent learns correct HSN/SAC → GST slab mappings |
| **Finance Teams** | GSTR-3B filing errors due to missed credits | End-to-end filing agent with error-flagging |

---

## 2. System Overview

### End-to-End Workflow

```
Raw Invoices (JSON batch)
        │
        ▼
 ┌─────────────────────┐
 │  Task 1: Classify   │  → GST category, HSN code, tax slab (0/5/12/18/28%)
 └─────────────────────┘
        │ score ∈ [0,1]
        ▼
 ┌─────────────────────┐
 │  Task 2: ITC Match  │  → Match GSTR-2A/2B vs Purchase Register, flag discrepancies
 └─────────────────────┘
        │ score ∈ [0,1]
        ▼
 ┌─────────────────────┐
 │  Task 3: GSTR-3B    │  → Compute liability, offset ITC, generate + submit return
 └─────────────────────┘
        │ score ∈ [0,1]
        ▼
  Composite Score + Audit Trail + stdout [STEP] logs
```

### Repository Structure (Claude Code Build Target)

```
gst_rl/                              ← root of HF Space repo
├── inference.py                     ← MANDATORY: hackathon entry point
├── openenv.yaml                     ← MANDATORY: openenv validate target
├── Dockerfile                       ← MANDATORY: containerized build
├── README.md                        ← MANDATORY: HF Space docs
├── requirements.txt
│
├── env/
│   ├── __init__.py
│   ├── gst_env.py                   ← Main Gymnasium Env (OpenEnv compliant)
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── task1_classify.py
│   │   ├── task2_itc.py
│   │   └── task3_filing.py
│   ├── graders/
│   │   ├── __init__.py
│   │   ├── grader_classify.py
│   │   ├── grader_itc.py
│   │   └── grader_filing.py
│   ├── rewards.py
│   └── memory_manager.py
│
├── models/
│   ├── __init__.py
│   ├── observation.py               ← Pydantic v2 GSTObservation
│   ├── action.py                    ← Pydantic v2 GSTAction
│   └── reward.py                    ← Pydantic v2 RewardSignal
│
├── data/
│   ├── synthetic/
│   │   ├── generator.py             ← Seeded synthetic invoice generator
│   │   └── hsn_table.json           ← HSN → slab lookup (500 entries)
│   ├── semantic/
│   │   └── gst_rules.json           ← GST law rules (static)
│   ├── train/                       ← 800 pre-generated episodes
│   ├── val/                         ← 100 episodes (held-out seed)
│   └── test/                        ← 100 curated episodes
│
└── agent/
    ├── baseline_agent.py            ← Deterministic rule-based baseline
    └── rl_agent.py                  ← PPO agent (stable-baselines3)
```

---

## 3. Hackathon Output Protocol (CRITICAL)

### inference.py Output Format

The script **must** emit exactly these line types to `stdout`. Any deviation causes scoring failure.

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

**Rules (verbatim from guidelines)**:
- One `[START]` line at episode begin
- One `[STEP]` line per step, **immediately after** `env.step()` returns
- One `[END]` line after `env.close()`, **always emitted** (even on exception)
- `reward` and `rewards` formatted to **2 decimal places**
- `done` and `success` are **lowercase booleans**: `true` or `false`
- `error` is the raw `last_action_error` string, or `null` if none
- All fields on a **single line** with no newlines within a line

**Example**:
```
[START] task=invoice_classification env=gst-intelligence model=gpt-4.1-mini
[STEP] step=1 action=classify_invoice(INV-001,B2B,8471,18) reward=0.85 done=false error=null
[STEP] step=2 action=classify_invoice(INV-002,B2C,6109,5) reward=0.90 done=false error=null
[STEP] step=3 action=classify_invoice(INV-003,RCM,9983,18) reward=1.00 done=true error=null
[END] success=true steps=3 rewards=0.85,0.90,1.00
```

### Environment Variables

```python
# inference.py — top of file, always
import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")  # MUST have default
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")               # MUST have default
HF_TOKEN     = os.getenv("HF_TOKEN")                                    # NO default (mandatory)

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
```

---

## 4. OpenEnv Interface Compliance

### `openenv.yaml`

```yaml
name: gst-intelligence-env
version: "1.0.0"
description: >
  RL environment for automated Indian GST workflows:
  invoice classification, ITC reconciliation, and GSTR-3B filing.
  Real-world task simulation for SME accounting automation.

interface:
  step:  "step(action: GSTAction) -> Tuple[GSTObservation, float, bool, bool, dict]"
  reset: "reset(task_id: int, seed: int) -> GSTObservation"
  state: "state() -> dict"

tasks:
  - id: 1
    name: invoice_classification
    difficulty: easy
    max_steps: 50
    score_range: [0.0, 1.0]
    description: >
      Classify each invoice with correct GST parameters:
      invoice type, HSN code, GST slab, ITC eligibility, and RCM flag.

  - id: 2
    name: itc_reconciliation
    difficulty: medium
    max_steps: 150
    score_range: [0.0, 1.0]
    description: >
      Match purchase register invoices against GSTR-2B entries.
      Flag discrepancies with correct type and recommended action.

  - id: 3
    name: gstr3b_filing
    difficulty: hard
    max_steps: 300
    score_range: [0.0, 1.0]
    description: >
      Complete end-to-end GSTR-3B return: compute outward liability,
      apply ITC reversals, offset credits, and submit the return.

constraints:
  max_vcpu: 2
  max_ram_gb: 8
  max_inference_minutes: 20

deployment:
  docker: true
  huggingface_spaces: true
  hf_tag: openenv                   # REQUIRED: space must be tagged openenv
  space_must_be_running: true       # Submission fails if space is stopped

reward:
  type: dense
  range: [-1.0, 1.0]
  partial_credit: true
  penalizes:
    - infinite_loops
    - invalid_actions
    - incorrect_filings
    - fake_itc_claims
```

### Core Python Interface

```python
# env/gst_env.py
import gymnasium as gym
from models.observation import GSTObservation
from models.action import GSTAction
from models.reward import RewardSignal
from env.rewards import RewardEngine
from env.memory_manager import MemoryManager

class GSTEnvironment(gym.Env):
    """
    OpenEnv-compliant GST workflow environment.
    Passes: openenv validate
    Tasks: invoice_classification (easy) → itc_reconciliation (medium) → gstr3b_filing (hard)
    """
    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.reward_engine = RewardEngine()
        self.memory = MemoryManager()
        self._state = {}

    def reset(self, *, seed: int = None, task_id: int = 1,
              options: dict = None) -> GSTObservation:
        """
        Load a new invoice batch. Returns initial GSTObservation.
        Called by inference.py at episode start.
        """
        super().reset(seed=seed)
        batch = self._load_batch(task_id, seed)
        self._state = self._init_state(batch, task_id)
        return self._build_observation()

    def step(self, action: GSTAction) -> tuple[GSTObservation, float, bool, bool, dict]:
        """
        Execute one agent action.
        Returns: (observation, reward, terminated, truncated, info)
        info["last_action_error"] used for [STEP] error field.
        """
        # Validate action legality
        if not self._is_valid_action(action):
            reward = -0.15
            info = {
                "last_action_error": f"invalid_action_{action.action_type}_in_{self._state['phase']}",
                "step": self._state["step"],
            }
            self._state["step"] += 1
            self._state["invalid_actions"] += 1
            return self._build_observation(), reward, False, False, info

        # Execute + compute reward
        reward_signal = self.reward_engine.compute(action, self._state)
        self._apply_action(action)
        self._state["step"] += 1
        self._state["cumulative_reward"] += reward_signal.step_reward

        terminated = self._check_terminal()
        truncated  = self._state["step"] >= self._max_steps()
        info = {
            "last_action_error": None,
            "step": self._state["step"],
            "phase": self._state["phase"],
            "sub_rewards": reward_signal.sub_rewards,
        }
        return self._build_observation(), reward_signal.step_reward, terminated, truncated, info

    def state(self) -> dict:
        """Full internal state for deterministic replay and audit."""
        return self._state.copy()

    def close(self):
        """Cleanup. [END] line must be printed AFTER this call."""
        pass
```

---

## 5. Observation Space (DETAILED)

### Pydantic Schemas

```python
# models/observation.py
from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum

class InvoiceStatus(str, Enum):
    PENDING    = "pending"
    CLASSIFIED = "classified"
    MATCHED    = "matched"
    FLAGGED    = "flagged"
    DISPUTED   = "disputed"

class InvoiceObservation(BaseModel):
    invoice_id:       str
    supplier_gstin:   Optional[str]          # None = missing (red flag)
    buyer_gstin:      str
    invoice_date:     str                    # "YYYY-MM-DD"
    invoice_type:     Literal["B2B","B2C","EXPORT","RCM","ISD"]
    hsn_code:         Optional[str]          # 4–8 digit HSN or SAC
    description:      str
    taxable_value:    float
    igst_amount:      float
    cgst_amount:      float
    sgst_amount:      float
    total_amount:     float
    gstr2b_match:     Optional[str]          # Matched GSTR-2B entry ID or None
    status:           InvoiceStatus
    flags:            list[str]              # ["missing_gstin","rate_mismatch","duplicate"]

class ITCMismatch(BaseModel):
    purchase_invoice_id: str
    gstr2b_invoice_id:   Optional[str]
    mismatch_type:       Literal[
        "amount_diff","gstin_missing","not_in_2b",
        "rate_mismatch","cancelled","duplicate"
    ]
    purchase_taxable:    float
    gstr2b_taxable:      Optional[float]
    delta:               float

class GSTR3BSummary(BaseModel):
    taxable_outward:     float = 0.0        # 3.1(a)
    zero_rated:          float = 0.0        # 3.1(b)
    exempted:            float = 0.0        # 3.1(c)
    rcm_inward:          float = 0.0        # 3.1(d)
    itc_igst:            float = 0.0        # 4(A)(5)
    itc_cgst:            float = 0.0
    itc_sgst:            float = 0.0
    itc_ineligible:      float = 0.0        # 4(B) reversals
    igst_payable:        float = 0.0        # 6.1
    cgst_payable:        float = 0.0        # 6.2
    sgst_payable:        float = 0.0        # 6.3
    net_payable:         float = 0.0
    sections_completed:  list[str] = []

class GSTObservation(BaseModel):
    episode_id:           str
    task_id:              int               # 1, 2, or 3
    step_number:          int
    gstin:                str
    tax_period:           str               # "2024-03"
    current_invoice:      Optional[InvoiceObservation]
    total_invoices:       int
    classified_count:     int
    pending_count:        int
    flagged_count:        int
    mismatches:           list[ITCMismatch]
    matched_itc_amount:   float
    disputed_itc_amount:  float
    gstr3b:               GSTR3BSummary
    steps_remaining:      int
    cumulative_reward:    float
    last_action_result:   Optional[str]     # "success"|"invalid"|"penalty"
    similar_past_invoices: list[dict]
    known_supplier_profile: Optional[dict]
```

### Example Observation (JSON)

```json
{
  "episode_id": "ep-20240315-042",
  "task_id": 2,
  "step_number": 34,
  "gstin": "27AABCU9603R1ZM",
  "tax_period": "2024-02",
  "current_invoice": {
    "invoice_id": "INV-2024-0234",
    "supplier_gstin": "29GGGGG1314R9Z6",
    "buyer_gstin": "27AABCU9603R1ZM",
    "invoice_date": "2024-02-14",
    "invoice_type": "B2B",
    "hsn_code": "8471",
    "description": "Laptop Computer - Core i7",
    "taxable_value": 75000.00,
    "igst_amount": 13500.00,
    "cgst_amount": 0.0,
    "sgst_amount": 0.0,
    "total_amount": 88500.00,
    "gstr2b_match": null,
    "status": "pending",
    "flags": ["not_in_2b"]
  },
  "total_invoices": 120,
  "classified_count": 98,
  "pending_count": 22,
  "flagged_count": 7,
  "mismatches": [
    {
      "purchase_invoice_id": "INV-2024-0198",
      "gstr2b_invoice_id": "2B-09823",
      "mismatch_type": "amount_diff",
      "purchase_taxable": 50000.0,
      "gstr2b_taxable": 48000.0,
      "delta": 2000.0
    }
  ],
  "matched_itc_amount": 425000.0,
  "disputed_itc_amount": 13500.0,
  "gstr3b": { "itc_igst": 425000.0, "sections_completed": ["3.1a"] },
  "steps_remaining": 116,
  "cumulative_reward": 0.43,
  "last_action_result": "success",
  "similar_past_invoices": [
    {"invoice_id": "INV-2024-0101", "hsn": "8471", "decision": "matched", "outcome": "accepted"}
  ],
  "known_supplier_profile": {
    "gstin": "29GGGGG1314R9Z6",
    "historical_compliance_rate": 0.94,
    "avg_delay_days": 2.1,
    "cancelled_invoices_pct": 0.02
  }
}
```

---

## 6. Action Space (DETAILED)

### Pydantic Schemas

```python
# models/action.py
from pydantic import BaseModel
from typing import Optional, Literal, Union
from enum import Enum

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
    invoice_type:   Literal["B2B","B2C","EXPORT","RCM","ISD","EXEMPT"]
    hsn_code:       str
    gst_slab:       Literal["0","5","12","18","28","exempt"]
    supply_type:    Literal["goods","services"]
    itc_eligible:   bool
    reverse_charge: bool

class MatchITCPayload(BaseModel):
    purchase_invoice_id: str
    gstr2b_invoice_id:   str
    confidence:          float

class FlagDiscrepancyPayload(BaseModel):
    invoice_id:          str
    discrepancy_type:    Literal[
        "amount_diff","gstin_missing","not_in_2b",
        "rate_mismatch","cancelled","duplicate","fake_invoice"
    ]
    recommended_action:  Literal["hold_itc","defer","dispute","write_off"]
    notes:               str

class SetSectionValuePayload(BaseModel):
    section: Literal[
        "3.1a","3.1b","3.1c","3.1d",
        "4a","4b","4c","4d",
        "6.1","6.2","6.3"
    ]
    value:   float

class GSTAction(BaseModel):
    action_type:      ActionType
    payload:          Union[
        ClassifyInvoicePayload, MatchITCPayload,
        FlagDiscrepancyPayload, SetSectionValuePayload, dict
    ]
    timestamp:        str
    agent_reasoning:  Optional[str] = None

    def to_action_str(self) -> str:
        """Compact string for [STEP] action= field."""
        p = self.payload
        if self.action_type == "classify_invoice":
            return f"classify_invoice({p.invoice_id},{p.invoice_type},{p.hsn_code},{p.gst_slab})"
        elif self.action_type == "match_itc":
            return f"match_itc({p.purchase_invoice_id},{p.gstr2b_invoice_id})"
        elif self.action_type == "flag_discrepancy":
            return f"flag_discrepancy({p.invoice_id},{p.discrepancy_type},{p.recommended_action})"
        elif self.action_type == "set_section_value":
            return f"set_section({p.section},{p.value:.2f})"
        else:
            return self.action_type
```

### Action Validity Map (State Machine)

```python
VALID_ACTIONS = {
    "CLASSIFYING": [
        "classify_invoice", "skip_invoice", "request_clarification"
    ],
    "RECONCILING": [
        "match_itc", "flag_discrepancy", "accept_mismatch",
        "defer_invoice", "skip_invoice"
    ],
    "FILING": [
        "set_section_value", "generate_return", "submit_return"
    ],
}
```

---

## 7. Reward Design (CRITICAL)

### Reward Schema

```python
# models/reward.py
from pydantic import BaseModel

class RewardSignal(BaseModel):
    step_reward:       float             # Immediate [-1, 1]
    cumulative_reward: float             # Episode total
    sub_rewards:       dict[str, float]  # Component breakdown
    penalty_flags:     list[str]         # Why penalties applied
    shaping_bonus:     float             # Potential-based shaping
```

### Task 1 — Invoice Classification

```
R_classify(t) =
    + 0.40 × [gst_slab correct]
    + 0.25 × [hsn_code correct]        → partial: 4-digit prefix match = 0.5×
    + 0.15 × [invoice_type correct]
    + 0.10 × [itc_eligible correct]
    + 0.10 × [reverse_charge correct]
    − 0.30 × [wrong slab]
    − 0.20 × [invalid/nonexistent HSN]
    − 0.10 × [skip_invoice used]
    + 0.05 × (steps_remaining / max_steps)    ← efficiency shaping
```

### Task 2 — ITC Reconciliation

```
R_itc(t) =
    + 0.50 × itc_match_score(purchase, gstr2b)    ← composite similarity [0,1]
    + 0.20 × [amount delta < INR 1000]
    + 0.15 × [discrepancy_type correct]
    + 0.15 × [recommended_action correct]
    − 0.40 × [false match: unrelated invoices]
    − 0.30 × [missed fraud flag]
    − 0.20 × [accepted large mismatch > INR 5000 without flagging]
    − 0.10 × [duplicate action on same invoice]
```

### Task 3 — GSTR-3B Filing

```
Per section entry:
    pct_error = |agent_value - true_value| / true_value
    R_section = +0.20  if pct_error < 0.05
                +0.10  if pct_error < 0.20
                −0.10  otherwise

Terminal reward on submit:
    + 0.50 × [all sections within 5% error]
    + 0.25 × [ITC correctly offset against liability]
    + 0.15 × [net_payable matches within 2%]
    + 0.10 × [submitted within step budget]
    − 0.50 × [unresolved discrepancies in filed return]
    − 0.30 × [net payable underpaid by > 2%]
    − 1.00 × [ITC overclaimed by > 2%]     ← episode-ending penalty
```

### Penalty Matrix

| Violation | Penalty |
|-----------|---------|
| Invalid action for current state | -0.15 |
| Repeated action on same invoice (loop) | -0.20 |
| False ITC match (different suppliers) | -0.40 |
| Filed return with unresolved discrepancies | -0.50 |
| RCM ITC claimed before paying tax | -0.30 |
| Fake invoice not flagged | -0.50 |

### Potential-Based Shaping (Anti-Reward-Hack)

```python
def shaped_reward(r_base: float, phi_s: float, phi_s_next: float, gamma=0.99) -> float:
    """F(s,s') = γΦ(s') - Φ(s) — guarantees policy invariance."""
    return r_base + gamma * phi_s_next - phi_s

def compute_potential(state: dict) -> float:
    correct_pct     = state["correct_decisions"] / max(state["total_decisions"], 1)
    itc_recovery    = state["matched_itc"] / max(state["eligible_itc"], 1)
    return 0.5 * correct_pct + 0.5 * itc_recovery
```

---

## 8. Task Definitions

### Task 1 (Easy): Invoice Classification

**Goal**: For each invoice in the batch, output: `invoice_type`, `hsn_code`, `gst_slab`, `supply_type`, `itc_eligible`, `reverse_charge`.

**Grader**:
```python
def grade_classification(pred, gt) -> float:
    weights = {"gst_slab": 0.40, "hsn_code": 0.25, "invoice_type": 0.15,
               "itc_eligible": 0.10, "reverse_charge": 0.10}
    score = 0.0
    score += weights["gst_slab"] if pred.gst_slab == gt.gst_slab else 0
    if pred.hsn_code == gt.hsn_code:
        score += weights["hsn_code"]
    elif pred.hsn_code[:4] == gt.hsn_code[:4]:
        score += weights["hsn_code"] * 0.5
    for f in ["invoice_type", "itc_eligible", "reverse_charge"]:
        if getattr(pred, f) == getattr(gt, f):
            score += weights[f]
    return round(score, 4)   # ∈ [0.0, 1.0]
```

---

### Task 2 (Medium): ITC Reconciliation

**Goal**: Match purchase register invoices to GSTR-2B; flag discrepancies with type + recommended action.

**Match Similarity**:
```python
def itc_match_score(purchase, gstr2b) -> float:
    gstin_score  = 1.0 if purchase.supplier_gstin == gstr2b.supplier_gstin else 0.0
    delta_pct    = abs(purchase.taxable_value - gstr2b.taxable_value) / (gstr2b.taxable_value + 1)
    amount_score = max(0, 1 - delta_pct / 0.10)
    date_diff    = abs((purchase.date - gstr2b.date).days)
    date_score   = max(0, 1 - date_diff / 30)
    num_sim      = fuzz.ratio(purchase.invoice_number, gstr2b.invoice_number) / 100
    return 0.35*gstin_score + 0.35*amount_score + 0.15*date_score + 0.15*num_sim
```

**Edge Cases**:

| Mismatch | Correct Action |
|----------|---------------|
| `not_in_2b` + high compliance supplier | `defer_invoice` |
| `amount_diff` < INR 1000 | `accept_mismatch` |
| `amount_diff` > INR 10K | `flag_discrepancy → dispute` |
| `cancelled` | `flag_discrepancy → hold_itc` |
| `fake_invoice` | `flag_discrepancy → dispute` |
| `duplicate` | `flag_discrepancy → write_off` |

---

### Task 3 (Hard): GSTR-3B Filing

**Goal**: Fill all GSTR-3B sections correctly and submit.

**Mandatory Section Sequence**:
```
3.1(a) → 3.1(b) → 3.1(c) → 3.1(d) →
4(A)(5) → 4(B) →
6.1 (IGST) → 6.2 (CGST) → 6.3 (SGST) →
submit_return
```

**ITC Offset Order** (legally mandated):
```
IGST ITC → offsets IGST first, then CGST, then SGST
CGST ITC → offsets CGST only
SGST ITC → offsets SGST only
```

**Terminal Grader**:
```python
def grade_gstr3b(agent_return, true_return) -> float:
    section_scores = {}
    for sec in GSTR3B_SECTIONS:
        a, t = getattr(agent_return, sec), getattr(true_return, sec)
        pct_err = abs(a - t) / (t + 1)
        section_scores[sec] = max(0.0, 1.0 - pct_err / 0.05)

    base = sum(section_scores.values()) / len(section_scores)
    underpay = -0.30 if agent_return.net_payable < true_return.net_payable * 0.98 else 0
    overclaim = -0.50 if agent_return.total_itc   > true_return.total_itc   * 1.02 else 0
    return round(max(0.0, min(1.0, base + underpay + overclaim)), 4)
```

---

## 9. Grader Design

```python
class BaseGrader(ABC):
    @abstractmethod
    def grade(self, prediction: dict, ground_truth: dict) -> float:
        """Returns deterministic score ∈ [0.0, 1.0]"""
        ...
    def is_deterministic(self) -> bool:
        return True

# Composite episode score
TASK_WEIGHTS = {1: 0.20, 2: 0.35, 3: 0.45}

def compute_episode_score(task_scores: dict[int, float]) -> float:
    return round(sum(TASK_WEIGHTS[t] * s for t, s in task_scores.items()), 4)
```

---

## 10. Edge Cases & Realism

| Edge Case | Prevalence | Correct Action |
|-----------|-----------|---------------|
| Fake invoice (invalid GSTIN checksum) | 5% | Flag as `fake_invoice` |
| Amount diff < INR 1000 | 15% | `accept_mismatch` |
| Missing GSTIN | 8% | Classify as B2C, no ITC |
| Wrong HSN from supplier | 12% | Correct from description |
| RCM not accounted | 10% | Add to 3.1(d), pay tax first |
| GSTR-2B late (supplier) | 20% | `defer_invoice` |
| Duplicate in purchase register | 3% | Flag as `duplicate` |
| Section 17(5) ineligible ITC | 10% | Reverse in 4(B) |
| Inter-state vs intra-state confusion | 8% | Check state codes in GSTIN |
| Cancelled post-claim | 5% | Reverse ITC |

---

## 11. Evaluation Metrics

| Metric | Target |
|--------|--------|
| Classification Accuracy | ≥ 0.90 |
| ITC Match Precision | ≥ 0.92 |
| ITC Match Recall | ≥ 0.88 |
| Filing Section Error | ≤ 5% |
| Fraud Detection Rate | ≥ 0.85 |
| Episode Completion Rate | ≥ 0.80 |
| Composite Compliance Score | ≥ 0.85 |

---

## 12. MVP Scope (Hackathon)

### Real
- Full OpenEnv interface (step / reset / state / close)
- Pydantic v2 typed models
- Deterministic graders
- Synthetic data generator with seeded reproducibility
- Rule-based deterministic baseline agent
- PPO training loop (stable-baselines3)
- Dense reward with penalties
- Docker + HF Spaces + `openenv` tag

### Simplified
- GSTIN validation: checksum-based mock (not live GSTN portal)
- GSTR-2B: JSON mock (not actual GSTN network)
- HSN table: 500-entry curated subset
- Single GSTIN per episode
- No amendment returns (GSTR-3BX)

### Data Split
| Split | Episodes | Seed |
|-------|----------|------|
| Train | 800 | 42 |
| Val   | 100 | 999 |
| Test  | 100 | manually curated |

---

## 13. Common Failure Cases (Avoid)

Per hackathon guidelines — these cause **automatic submission failure**:

1. `inference.py` not in root directory
2. Missing default values for `API_BASE_URL` or `MODEL_NAME`
3. `HF_TOKEN` missing raises no error (it must raise `ValueError`)
4. HuggingFace Space still building during submission
5. Space stopped due to multiple active deployments (turn off other spaces)
6. `[END]` line not emitted on exception (use `try/finally`)
7. `openenv validate` not passing (run locally before submit)
8. Space not tagged with `openenv`
