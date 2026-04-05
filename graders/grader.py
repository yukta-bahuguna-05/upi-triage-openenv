"""
grader.py — Upgraded deterministic grader for UPI Triage Environment v2.

New scoring components:
  - consistency_score: penalizes contradictory decisions on same merchant
  - action_diversity_score: rewards using richer action types
  - deferred_resolved: tracks deferred transaction handling

Final score weights:
  0.50 × category_score
  0.20 × fraud_recall
  0.10 × duplicate_accuracy
  0.10 × consistency_score
  0.05 × action_diversity_score
  0.05 × (1 - false_positive_rate)
"""

from env.models import (
    GradeResult, Category, FlagType, ActionType,
    Difficulty, UPITransaction, AgentAction
)
from typing import List, Dict, Optional


CLOSE_CATEGORIES = {
    Category.FOOD_DELIVERY:  [Category.GROCERIES, Category.OTHER],
    Category.GROCERIES:      [Category.FOOD_DELIVERY, Category.OTHER],
    Category.TRANSPORT:      [Category.OTHER],
    Category.ENTERTAINMENT:  [Category.SUBSCRIPTION, Category.OTHER],
    Category.SUBSCRIPTION:   [Category.ENTERTAINMENT, Category.UTILITIES],
    Category.UTILITIES:      [Category.SUBSCRIPTION, Category.OTHER],
    Category.RENT:           [Category.TRANSFER, Category.OTHER],
    Category.TRANSFER:       [Category.RENT, Category.BILL_SPLIT],
    Category.BILL_SPLIT:     [Category.TRANSFER, Category.OTHER],
    Category.SHOPPING:       [Category.OTHER],
    Category.MEDICAL:        [Category.OTHER],
    Category.EDUCATION:      [Category.SUBSCRIPTION, Category.OTHER],
    Category.INVESTMENT:     [Category.OTHER],
    Category.SALARY:         [Category.TRANSFER, Category.OTHER],
    Category.REFUND:         [Category.OTHER],
    Category.SUSPICIOUS:     [Category.OTHER],
    Category.OTHER:          [],
}


def grade(
    transactions: List[UPITransaction],
    actions: List[AgentAction],
    difficulty: str,
    action_type_counts: Optional[Dict[str, int]] = None,
) -> GradeResult:
    """
    Deterministically grade a full episode.

    Args:
        transactions: full list of transactions processed
        actions: agent's actions in order
        difficulty: easy/medium/hard
        action_type_counts: how many of each action type were used

    Returns:
        GradeResult with score 0.0 to 1.0 and full breakdown
    """
    assert len(transactions) == len(actions), \
        "Transactions and actions must have same length"

    total = len(transactions)
    if total == 0:
        return GradeResult(
            score=0.0, category_accuracy=0.0, flag_accuracy=0.0,
            fraud_recall=0.0, false_positive_rate=0.0,
            duplicate_accuracy=0.0, consistency_score=1.0,
            action_diversity_score=0.0, total_transactions=0,
            correct_categories=0, correct_flags=0,
            deferred_resolved=0, info_requests_used=0,
            difficulty=Difficulty(difficulty), breakdown={}
        )

    # ── Category scoring ──────────────────────────────────────
    exact_correct = 0
    partial_correct = 0
    category_breakdown = {}

    for txn, action in zip(transactions, actions):
        cat_key = txn.ground_truth_category.value
        if cat_key not in category_breakdown:
            category_breakdown[cat_key] = {"total": 0, "correct": 0, "partial": 0}
        category_breakdown[cat_key]["total"] += 1

        if action.action_type == ActionType.CATEGORIZE and action.category:
            if action.category == txn.ground_truth_category:
                exact_correct += 1
                category_breakdown[cat_key]["correct"] += 1
            elif action.category in CLOSE_CATEGORIES.get(txn.ground_truth_category, []):
                partial_correct += 1
                category_breakdown[cat_key]["partial"] += 1

    category_score_raw = (exact_correct + 0.5 * partial_correct) / total
    category_accuracy = round(exact_correct / total, 4)

    # ── Fraud detection ───────────────────────────────────────
    fraud_pairs = [
        (txn, action) for txn, action in zip(transactions, actions)
        if txn.ground_truth_flag == FlagType.SUSPICIOUS
    ]
    total_fraud = len(fraud_pairs)
    fraud_caught = sum(
        1 for txn, action in fraud_pairs
        if action.flag == FlagType.SUSPICIOUS or
           action.action_type in [ActionType.DISPUTE, ActionType.MARK_DUPLICATE]
    )
    fraud_recall = round(fraud_caught / total_fraud, 4) if total_fraud > 0 else 1.0

    # ── Duplicate detection ───────────────────────────────────
    dup_pairs = [
        (txn, action) for txn, action in zip(transactions, actions)
        if txn.ground_truth_flag == FlagType.DUPLICATE
    ]
    total_dups = len(dup_pairs)
    dups_caught = sum(
        1 for txn, action in dup_pairs
        if action.flag == FlagType.DUPLICATE or
           action.action_type == ActionType.MARK_DUPLICATE
    )
    duplicate_accuracy = round(dups_caught / total_dups, 4) if total_dups > 0 else 1.0

    # ── False positives ───────────────────────────────────────
    normal_pairs = [
        (txn, action) for txn, action in zip(transactions, actions)
        if txn.ground_truth_flag == FlagType.NORMAL
    ]
    total_normal = len(normal_pairs)
    false_positives = sum(
        1 for txn, action in normal_pairs
        if action.flag == FlagType.SUSPICIOUS and
           action.action_type != ActionType.DISPUTE
    )
    false_positive_rate = round(false_positives / total_normal, 4) if total_normal > 0 else 0.0

    # ── Flag accuracy overall ─────────────────────────────────
    correct_flags = sum(
        1 for txn, action in zip(transactions, actions)
        if action.flag == txn.ground_truth_flag
    )
    flag_accuracy = round(correct_flags / total, 4)

    # ── Consistency score ─────────────────────────────────────
    # Check if agent was consistent across same merchant
    merchant_decisions: Dict[str, List[Category]] = {}
    for txn, action in zip(transactions, actions):
        if action.category:
            key = txn.upi_id
            if key not in merchant_decisions:
                merchant_decisions[key] = []
            merchant_decisions[key].append(action.category)

    inconsistent_merchants = 0
    total_multi_merchants = 0
    for upi_id, cats in merchant_decisions.items():
        if len(cats) > 1:
            total_multi_merchants += 1
            if len(set(cats)) > 1:  # More than one unique category for same merchant
                inconsistent_merchants += 1

    if total_multi_merchants > 0:
        consistency_score = round(
            1.0 - (inconsistent_merchants / total_multi_merchants), 4
        )
    else:
        consistency_score = 1.0

    # ── Action diversity score ────────────────────────────────
    counts = action_type_counts or {}
    diverse_used = sum(
        1 for at in ActionType
        if at != ActionType.CATEGORIZE and counts.get(at.value, 0) > 0
    )
    action_diversity_score = round(min(diverse_used / 3.0, 1.0), 4)

    # ── Info requests and deferred ────────────────────────────
    info_requests_used = counts.get(ActionType.REQUEST_INFO.value, 0)
    deferred_resolved = counts.get(ActionType.DEFER.value, 0)

    # ── Final weighted score ──────────────────────────────────
    score = (
        0.50 * category_score_raw +
        0.20 * fraud_recall +
        0.10 * duplicate_accuracy +
        0.10 * consistency_score +
        0.05 * action_diversity_score +
        0.05 * (1.0 - false_positive_rate)
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    return GradeResult(
        score=score,
        category_accuracy=category_accuracy,
        flag_accuracy=flag_accuracy,
        fraud_recall=fraud_recall,
        false_positive_rate=false_positive_rate,
        duplicate_accuracy=duplicate_accuracy,
        consistency_score=consistency_score,
        action_diversity_score=action_diversity_score,
        total_transactions=total,
        correct_categories=exact_correct,
        correct_flags=correct_flags,
        deferred_resolved=deferred_resolved,
        info_requests_used=info_requests_used,
        difficulty=Difficulty(difficulty),
        breakdown=category_breakdown,
    )
