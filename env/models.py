"""
models.py — All typed data models for the UPI Triage Environment.

v2 changes:
  - Richer ActionType enum (not just classify + flag)
  - Observation now includes transaction history, trends, velocity
  - Agent decisions affect future state
  - Consistency tracking added
"""

from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────

class Category(str, Enum):
    FOOD_DELIVERY   = "food_delivery"
    GROCERIES       = "groceries"
    TRANSPORT       = "transport"
    ENTERTAINMENT   = "entertainment"
    UTILITIES       = "utilities"
    RENT            = "rent"
    SHOPPING        = "shopping"
    MEDICAL         = "medical"
    EDUCATION       = "education"
    INVESTMENT      = "investment"
    SUBSCRIPTION    = "subscription"
    TRANSFER        = "transfer"
    BILL_SPLIT      = "bill_split"
    SALARY          = "salary"
    REFUND          = "refund"
    SUSPICIOUS      = "suspicious"
    OTHER           = "other"


class FlagType(str, Enum):
    NORMAL          = "normal"
    SUSPICIOUS      = "suspicious"
    DUPLICATE       = "duplicate"
    NEEDS_REVIEW    = "needs_review"


class TransactionType(str, Enum):
    DEBIT   = "debit"
    CREDIT  = "credit"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ─────────────────────────────────────────────────────────────
# ACTION TYPE — what kind of decision the agent makes
# Fix for "too simple" action space
# ─────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """
    The type of action the agent takes on a transaction.

    CATEGORIZE       — standard classify + flag decision
    REQUEST_INFO     — agent needs more context before deciding
                       costs 1 step, returns enriched observation
    DEFER            — skip for now, revisit at end of episode
    MARK_DUPLICATE   — explicitly mark as duplicate of a previous txn
    DISPUTE          — flag for human review + freeze future txns from merchant
    UPDATE_PREVIOUS  — correct a previous categorization decision
    SUMMARIZE        — trigger intermediate budget summary update
    """
    CATEGORIZE      = "categorize"
    REQUEST_INFO    = "request_info"
    DEFER           = "defer"
    MARK_DUPLICATE  = "mark_duplicate"
    DISPUTE         = "dispute"
    UPDATE_PREVIOUS = "update_previous"
    SUMMARIZE       = "summarize"


# ─────────────────────────────────────────────────────────────
# TRANSACTION
# ─────────────────────────────────────────────────────────────

class UPITransaction(BaseModel):
    id: str
    amount: float
    merchant_name: str
    upi_id: str
    timestamp: str
    transaction_type: TransactionType
    description: Optional[str] = None
    bank_ref: str

    # Ground truth — hidden from agent
    ground_truth_category: Category
    ground_truth_flag: FlagType
    is_recurring: bool = False

    # Extra context — revealed only if agent uses REQUEST_INFO
    extra_context: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# TRANSACTION SUMMARY — compact view of a past transaction
# Included in history window in Observation
# ─────────────────────────────────────────────────────────────

class TransactionSummary(BaseModel):
    """Compact view of a past transaction shown in history window."""
    id: str
    amount: float
    merchant_name: str
    timestamp: str
    transaction_type: TransactionType
    agent_category: Optional[Category] = None
    agent_flag: Optional[FlagType] = None


# ─────────────────────────────────────────────────────────────
# SPENDING TREND — per-category trend data
# ─────────────────────────────────────────────────────────────

class SpendingTrend(BaseModel):
    category: str
    total_spent: float
    transaction_count: int
    average_amount: float
    last_seen: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# MERCHANT PATTERN — frequency of a merchant
# ─────────────────────────────────────────────────────────────

class MerchantPattern(BaseModel):
    merchant_name: str
    upi_id: str
    count: int
    total_amount: float
    average_amount: float
    agent_categories_used: List[str]


# ─────────────────────────────────────────────────────────────
# ACTION — richer than just category + flag
# ─────────────────────────────────────────────────────────────

class AgentAction(BaseModel):
    """
    The agent's decision on the current transaction.

    action_type determines required fields:
      CATEGORIZE      -> category + flag required
      REQUEST_INFO    -> no other fields (costs a step, returns extra context)
      DEFER           -> no other fields (revisited at end)
      MARK_DUPLICATE  -> duplicate_of_id required
      DISPUTE         -> flag=suspicious, freezes merchant
      UPDATE_PREVIOUS -> target_id + new_category required
      SUMMARIZE       -> no other fields (triggers summary update)
    """
    transaction_id: str
    action_type: ActionType = ActionType.CATEGORIZE

    # For CATEGORIZE and DISPUTE
    category: Optional[Category] = None
    flag: Optional[FlagType] = None

    # For MARK_DUPLICATE
    duplicate_of_id: Optional[str] = None

    # For UPDATE_PREVIOUS
    target_id: Optional[str] = None
    new_category: Optional[Category] = None
    new_flag: Optional[FlagType] = None

    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# OBSERVATION — now with history + trends + velocity
# ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Everything the agent sees at current step.

    v2 adds sequential intelligence:
      - recent_history: last 5 transactions with agent decisions
      - spending_trends: per-category running totals and averages
      - merchant_patterns: frequency map of merchants seen
      - spending_velocity_24h: total spent in last 24 hours
      - time_since_last_txn: seconds since last transaction
      - deferred_count: pending deferred transactions
      - disputed_merchants: frozen merchants
      - info_context: extra info after REQUEST_INFO
    """
    # Current transaction
    transaction_id: str
    amount: float
    merchant_name: str
    upi_id: str
    timestamp: str
    transaction_type: TransactionType
    description: Optional[str]
    bank_ref: str

    # Sequential intelligence
    recent_history: List[TransactionSummary] = Field(default_factory=list)
    spending_trends: List[SpendingTrend] = Field(default_factory=list)
    merchant_patterns: List[MerchantPattern] = Field(default_factory=list)
    spending_velocity_24h: float = 0.0
    time_since_last_txn: Optional[float] = None
    deferred_count: int = 0
    disputed_merchants: List[str] = Field(default_factory=list)
    info_context: Optional[str] = None
    is_deferred: bool = False

    # Episode context
    step: int
    total_steps: int
    transactions_remaining: int
    total_spent: float
    total_received: float
    category_counts: dict
    flagged_count: int
    difficulty: Difficulty


# ─────────────────────────────────────────────────────────────
# STEP RESULT
# ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict


# ─────────────────────────────────────────────────────────────
# GRADE RESULT
# ─────────────────────────────────────────────────────────────

class GradeResult(BaseModel):
    score: float
    category_accuracy: float
    flag_accuracy: float
    fraud_recall: float
    false_positive_rate: float
    duplicate_accuracy: float
    consistency_score: float
    action_diversity_score: float
    total_transactions: int
    correct_categories: int
    correct_flags: int
    deferred_resolved: int
    info_requests_used: int
    difficulty: Difficulty
    breakdown: dict
