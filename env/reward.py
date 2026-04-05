"""
reward.py — Upgraded dense reward function for UPI Triage Environment.

v2 changes:
  - Penalties for inconsistent decisions on same merchant
  - Rewards for using richer action types appropriately
  - Delayed reward component for accurate budget summary
  - Rewards for detecting spending patterns (velocity spikes)
  - Penalties for overusing REQUEST_INFO or DEFER
"""

from env.models import (
    Category, FlagType, ActionType, AgentAction,
    UPITransaction
)
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# CATEGORY SIMILARITY — for partial credit
# ─────────────────────────────────────────────────────────────

CATEGORY_NEIGHBORS = {
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


def compute_reward(
    action: AgentAction,
    transaction: UPITransaction,
    merchant_history: Dict[str, List[Category]],   # past decisions per merchant
    action_counts: Dict[str, int],                  # how many of each action type
    spending_velocity_24h: float,                   # total spent in last 24h
    episode_step: int,
) -> tuple[float, dict]:
    """
    Compute reward for a single agent action.

    Now accounts for:
      1. Category correctness (with partial credit)
      2. Flag correctness (fraud/duplicate detection)
      3. Action type appropriateness
      4. Consistency with past decisions on same merchant
      5. Confidence calibration
      6. Overuse penalties (too many defers/requests)
    """

    reward = 0.0
    info = {
        "action_type": action.action_type.value,
        "correct_category": False,
        "partial_category": False,
        "correct_flag": False,
        "fraud_missed": False,
        "false_alarm": False,
        "duplicate_caught": False,
        "consistency_penalty": 0.0,
        "action_type_reward": 0.0,
        "category_reward": 0.0,
        "flag_reward": 0.0,
        "ground_truth_category": transaction.ground_truth_category,
        "ground_truth_flag": transaction.ground_truth_flag,
    }

    # ─────────────────────────────────────────────────────────
    # Handle non-CATEGORIZE action types first
    # ─────────────────────────────────────────────────────────

    if action.action_type == ActionType.REQUEST_INFO:
        # Good if transaction is genuinely ambiguous (cryptic name)
        is_cryptic = any(c.isdigit() for c in transaction.merchant_name) or \
                     len(transaction.merchant_name) < 5 or \
                     "/" in transaction.merchant_name or \
                     "*" in transaction.merchant_name
        if is_cryptic:
            reward = 0.5   # Good use of REQUEST_INFO on hard transaction
            info["action_type_reward"] = 0.5
            info["note"] = "Appropriate use of REQUEST_INFO on cryptic transaction"
        else:
            reward = -0.3  # Unnecessary — merchant name was clear
            info["action_type_reward"] = -0.3
            info["note"] = "Unnecessary REQUEST_INFO — merchant was clear"

        # Overuse penalty — if agent uses REQUEST_INFO too much
        ri_count = action_counts.get(ActionType.REQUEST_INFO.value, 0)
        if ri_count > 5:
            overuse_penalty = -0.2 * (ri_count - 5)
            reward += overuse_penalty
            info["overuse_penalty"] = overuse_penalty

        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    if action.action_type == ActionType.DEFER:
        # Good if transaction is suspicious or needs pattern context
        is_suspicious = transaction.ground_truth_flag == FlagType.SUSPICIOUS
        if is_suspicious:
            reward = 0.3   # Reasonable to defer suspicious transaction
            info["action_type_reward"] = 0.3
        else:
            reward = -0.2  # Deferring normal transactions wastes steps
            info["action_type_reward"] = -0.2

        # Heavy overuse penalty
        defer_count = action_counts.get(ActionType.DEFER.value, 0)
        if defer_count > 3:
            overuse_penalty = -0.5 * (defer_count - 3)
            reward += overuse_penalty
            info["overuse_penalty"] = overuse_penalty

        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    if action.action_type == ActionType.SUMMARIZE:
        # Reward for triggering summary at meaningful checkpoints
        # Good at step 20, 40 in medium/hard tasks
        if episode_step in [19, 39]:
            reward = 0.5
            info["action_type_reward"] = 0.5
            info["note"] = "Good checkpoint summary"
        else:
            reward = 0.0   # Neutral — not wrong, just not optimal
        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    if action.action_type == ActionType.DISPUTE:
        # Agent is disputing this merchant — freezes future txns
        if transaction.ground_truth_flag == FlagType.SUSPICIOUS:
            reward = 2.5   # Great — correctly disputed a fraud merchant
            info["action_type_reward"] = 2.5
            info["correct_flag"] = True
        else:
            reward = -1.0  # False dispute — disrupts normal merchant
            info["action_type_reward"] = -1.0
            info["false_alarm"] = True
        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    if action.action_type == ActionType.UPDATE_PREVIOUS:
        # Agent is correcting a past decision
        # We reward this if the correction moves toward ground truth
        # (checked in upi_env.py where past decisions are accessible)
        reward = 0.2   # Small reward for self-correction behavior
        info["action_type_reward"] = 0.2
        info["note"] = "Agent corrected a previous decision"
        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    if action.action_type == ActionType.MARK_DUPLICATE:
        if transaction.ground_truth_flag == FlagType.DUPLICATE:
            reward = 1.8   # Correctly identified duplicate
            info["duplicate_caught"] = True
            info["correct_flag"] = True
        else:
            reward = -0.5  # False duplicate claim
            info["false_alarm"] = True
        info["action_type_reward"] = reward
        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    # ─────────────────────────────────────────────────────────
    # CATEGORIZE action — main path
    # ─────────────────────────────────────────────────────────

    if action.category is None or action.flag is None:
        # Agent forgot to provide category/flag for CATEGORIZE action
        reward = -0.5
        info["note"] = "CATEGORIZE action missing category or flag"
        info["total_reward"] = round(reward, 2)
        return round(reward, 2), info

    # Part 1 — Category reward
    if action.category == transaction.ground_truth_category:
        category_reward = 1.0
        info["correct_category"] = True
    elif action.category in CATEGORY_NEIGHBORS.get(transaction.ground_truth_category, []):
        category_reward = 0.5
        info["partial_category"] = True
    else:
        category_reward = 0.0

    info["category_reward"] = category_reward
    reward += category_reward

    # Part 2 — Consistency check
    # If agent previously categorized same merchant differently → penalty
    merchant_key = transaction.upi_id
    past_decisions = merchant_history.get(merchant_key, [])
    if past_decisions and action.category not in past_decisions:
        # Agent is being inconsistent about this merchant
        consistency_penalty = -0.3
        reward += consistency_penalty
        info["consistency_penalty"] = consistency_penalty
        info["note"] = f"Inconsistent: previously used {[c.value for c in past_decisions]}"

    # Part 3 — Flag reward
    ground_flag = transaction.ground_truth_flag
    agent_flag = action.flag

    if ground_flag == FlagType.SUSPICIOUS:
        if agent_flag == FlagType.SUSPICIOUS:
            flag_reward = 2.0
            info["correct_flag"] = True
        elif agent_flag == FlagType.NEEDS_REVIEW:
            flag_reward = 0.5
        else:
            flag_reward = -1.0
            info["fraud_missed"] = True

    elif ground_flag == FlagType.DUPLICATE:
        if agent_flag == FlagType.DUPLICATE:
            flag_reward = 1.5
            info["correct_flag"] = True
            info["duplicate_caught"] = True
        elif agent_flag == FlagType.NEEDS_REVIEW:
            flag_reward = 0.5
        else:
            flag_reward = -0.5

    elif ground_flag == FlagType.NORMAL:
        if agent_flag == FlagType.NORMAL:
            flag_reward = 0.3
            info["correct_flag"] = True
        elif agent_flag == FlagType.SUSPICIOUS:
            flag_reward = -0.5
            info["false_alarm"] = True
        else:
            flag_reward = -0.1

    elif ground_flag == FlagType.NEEDS_REVIEW:
        if agent_flag == FlagType.NEEDS_REVIEW:
            flag_reward = 0.5
            info["correct_flag"] = True
        elif agent_flag == FlagType.SUSPICIOUS:
            flag_reward = 0.3
        else:
            flag_reward = 0.0
    else:
        flag_reward = 0.0

    info["flag_reward"] = flag_reward
    reward += flag_reward

    # Part 4 — Spending velocity reward
    # Reward agent for flagging suspicious transactions when velocity is high
    if spending_velocity_24h > 10000 and agent_flag == FlagType.SUSPICIOUS \
            and ground_flag == FlagType.SUSPICIOUS:
        velocity_bonus = 0.5
        reward += velocity_bonus
        info["velocity_bonus"] = velocity_bonus

    # Part 5 — Confidence calibration
    if action.confidence is not None:
        if info["correct_category"] and action.confidence >= 0.8:
            reward += 0.1
        elif not info["correct_category"] and action.confidence >= 0.8:
            reward -= 0.1

    reward = round(reward, 2)
    info["total_reward"] = reward
    return reward, info


def compute_episode_bonus(
    total_correct_categories: int,
    total_transactions: int,
    fraud_caught: int,
    total_fraud: int,
    false_positives: int,
    spending_trends: Dict[str, float],    # final per-category totals
    ground_truth_trends: Dict[str, float], # correct per-category totals
    action_type_counts: Dict[str, int],   # how many of each action used
) -> tuple[float, dict]:
    """
    End of episode bonus rewards.

    v2 adds:
      - Budget summary accuracy bonus (delayed reward)
      - Action diversity bonus (used richer actions appropriately)
      - Consistency bonus (no contradictory decisions)
    """
    bonus = 0.0
    info = {}

    # Accuracy bonus
    accuracy = total_correct_categories / total_transactions if total_transactions > 0 else 0
    if accuracy >= 0.9:
        bonus += 5.0
        info["accuracy_bonus"] = 5.0
    elif accuracy >= 0.8:
        bonus += 3.0
        info["accuracy_bonus"] = 3.0
    elif accuracy >= 0.7:
        bonus += 1.0
        info["accuracy_bonus"] = 1.0
    else:
        info["accuracy_bonus"] = 0.0

    # Fraud recall bonus
    if total_fraud > 0:
        fraud_recall = fraud_caught / total_fraud
        if fraud_recall == 1.0:
            bonus += 3.0
            info["fraud_bonus"] = 3.0
        elif fraud_recall >= 0.5:
            bonus += 1.0
            info["fraud_bonus"] = 1.0
        else:
            info["fraud_bonus"] = 0.0

    # Zero false positives bonus
    if false_positives == 0:
        bonus += 2.0
        info["no_false_positive_bonus"] = 2.0
    else:
        info["no_false_positive_bonus"] = 0.0

    # Delayed reward — budget summary accuracy
    # Compare agent's category totals vs ground truth totals
    summary_error = 0.0
    for cat, true_amount in ground_truth_trends.items():
        agent_amount = spending_trends.get(cat, 0.0)
        if true_amount > 0:
            summary_error += abs(agent_amount - true_amount) / true_amount
    avg_error = summary_error / len(ground_truth_trends) if ground_truth_trends else 0
    if avg_error < 0.1:
        summary_bonus = 3.0
    elif avg_error < 0.25:
        summary_bonus = 1.5
    elif avg_error < 0.5:
        summary_bonus = 0.5
    else:
        summary_bonus = 0.0
    bonus += summary_bonus
    info["summary_accuracy_bonus"] = summary_bonus

    # Action diversity bonus — reward for using richer actions
    diverse_actions_used = sum(
        1 for action_type, count in action_type_counts.items()
        if action_type != ActionType.CATEGORIZE.value and count > 0
    )
    if diverse_actions_used >= 3:
        diversity_bonus = 2.0
    elif diverse_actions_used >= 2:
        diversity_bonus = 1.0
    elif diverse_actions_used >= 1:
        diversity_bonus = 0.5
    else:
        diversity_bonus = 0.0
    bonus += diversity_bonus
    info["diversity_bonus"] = diversity_bonus

    info["total_bonus"] = round(bonus, 2)
    return round(bonus, 2), info
