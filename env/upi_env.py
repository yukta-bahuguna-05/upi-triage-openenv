"""
upi_env.py — Upgraded UPI Triage Environment v2.

v2 changes:
  - Agent decisions affect future state
  - DISPUTE action freezes merchant (future txns pre-flagged)
  - DEFER action queues transactions for later revisit
  - REQUEST_INFO reveals extra context on next step
  - UPDATE_PREVIOUS allows correcting past decisions
  - Sequential intelligence: history, trends, velocity, patterns
  - Consistency tracking across merchant decisions
"""

from typing import Optional, List, Dict
from datetime import datetime
from env.models import (
    UPITransaction, AgentAction, Observation, StepResult,
    Category, FlagType, ActionType, Difficulty, TransactionType,
    TransactionSummary, SpendingTrend, MerchantPattern
)
from env.reward import compute_reward, compute_episode_bonus
from tasks.generator import load_transactions


class UPITriageEnv:
    """
    UPI Transaction Triage Environment v2.

    Agent decisions now affect future state:
      - DISPUTE merchant → future txns from that merchant pre-flagged
      - DEFER transaction → reappears at end of episode
      - REQUEST_INFO → next observation has extra_context filled
      - UPDATE_PREVIOUS → corrects past decision, adjusts running totals
    """

    def __init__(self, difficulty: str = "easy"):
        self.difficulty = Difficulty(difficulty)
        self.transactions: List[UPITransaction] = []
        self.current_step: int = 0
        self.done: bool = False

        # Agent decision tracking
        self.actions_taken: List[AgentAction] = []
        self.rewards: List[float] = []
        self.step_infos: List[dict] = []

        # Running financial counters
        self.total_spent: float = 0.0
        self.total_received: float = 0.0
        self.category_counts: Dict[str, int] = {}
        self.category_totals: Dict[str, float] = {}  # For budget summary
        self.flagged_count: int = 0

        # Sequential intelligence state
        self.recent_history: List[TransactionSummary] = []
        self.merchant_history: Dict[str, List[Category]] = {}  # upi_id → [categories]
        self.merchant_amounts: Dict[str, List[float]] = {}     # upi_id → [amounts]
        self.merchant_counts: Dict[str, int] = {}
        self.last_timestamp: Optional[str] = None
        self.spending_last_24h: List[tuple] = []  # (timestamp, amount)

        # State-affecting agent decisions
        self.disputed_merchants: List[str] = []    # Frozen merchants
        self.deferred_transactions: List[UPITransaction] = []  # Queued for later
        self.pending_info_request: bool = False    # Next step shows extra context
        self.action_type_counts: Dict[str, int] = {}

        # For grading
        self.final_transactions: List[UPITransaction] = []
        self.final_actions: List[AgentAction] = []

    # ─────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.transactions = load_transactions(self.difficulty.value)
        self.current_step = 0
        self.done = False
        self.actions_taken = []
        self.rewards = []
        self.step_infos = []
        self.total_spent = 0.0
        self.total_received = 0.0
        self.category_counts = {}
        self.category_totals = {}
        self.flagged_count = 0
        self.recent_history = []
        self.merchant_history = {}
        self.merchant_amounts = {}
        self.merchant_counts = {}
        self.last_timestamp = None
        self.spending_last_24h = []
        self.disputed_merchants = []
        self.deferred_transactions = []
        self.pending_info_request = False
        self.action_type_counts = {}
        self.final_transactions = []
        self.final_actions = []
        return self._make_observation()

    # ─────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────

    def step(self, action: AgentAction) -> StepResult:
        if self.done:
            raise ValueError("Episode done. Call reset().")

        current_txn = self.transactions[self.current_step]

        if action.transaction_id != current_txn.id:
            raise ValueError(
                f"Action transaction_id '{action.transaction_id}' "
                f"does not match current '{current_txn.id}'"
            )

        # Track action type counts
        at = action.action_type.value
        self.action_type_counts[at] = self.action_type_counts.get(at, 0) + 1

        # ── Handle state-affecting actions ────────────────────

        info_context = None

        if action.action_type == ActionType.REQUEST_INFO:
            # Don't advance — return same transaction with extra context
            self.pending_info_request = True
            reward = 0.5 if self._is_cryptic(current_txn) else -0.3
            info = {
                "action_type": "request_info",
                "note": "Returning enriched observation with extra context",
                "total_reward": reward,
            }
            obs = self._make_observation(info_context=current_txn.extra_context)
            self.rewards.append(reward)
            self.step_infos.append(info)
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        if action.action_type == ActionType.DEFER:
            # Queue transaction for later, move to next
            self.deferred_transactions.append(current_txn)
            reward = 0.3 if current_txn.ground_truth_flag == FlagType.SUSPICIOUS else -0.2
            info = {
                "action_type": "defer",
                "note": f"Deferred {current_txn.id}. {len(self.deferred_transactions)} deferred total.",
                "total_reward": reward,
            }
            self.rewards.append(reward)
            self.step_infos.append(info)
            self.current_step += 1
            return self._check_done_or_next(reward, info)

        if action.action_type == ActionType.DISPUTE:
            # Freeze this merchant for future transactions
            if current_txn.upi_id not in self.disputed_merchants:
                self.disputed_merchants.append(current_txn.upi_id)
            reward = 2.5 if current_txn.ground_truth_flag == FlagType.SUSPICIOUS else -1.0
            info = {
                "action_type": "dispute",
                "merchant_frozen": current_txn.upi_id,
                "note": f"Merchant {current_txn.merchant_name} disputed and frozen",
                "total_reward": reward,
            }
            self._update_history(current_txn, action)
            self.rewards.append(reward)
            self.step_infos.append(info)
            self.final_transactions.append(current_txn)
            self.final_actions.append(action)
            self.current_step += 1
            return self._check_done_or_next(reward, info)

        if action.action_type == ActionType.UPDATE_PREVIOUS:
            # Correct a past decision — no step advance
            reward = 0.2
            info = {
                "action_type": "update_previous",
                "target_id": action.target_id,
                "new_category": action.new_category.value if action.new_category else None,
                "total_reward": reward,
            }
            # Update the category counts if category changed
            if action.new_category and action.target_id:
                old_action = next(
                    (a for a in self.actions_taken if a.transaction_id == action.target_id),
                    None
                )
                if old_action and old_action.category:
                    old_cat = old_action.category.value
                    new_cat = action.new_category.value
                    if self.category_counts.get(old_cat, 0) > 0:
                        self.category_counts[old_cat] -= 1
                    self.category_counts[new_cat] = self.category_counts.get(new_cat, 0) + 1
            self.rewards.append(reward)
            self.step_infos.append(info)
            self.actions_taken.append(action)
            obs = self._make_observation()
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        if action.action_type == ActionType.SUMMARIZE:
            reward = 0.5 if self.current_step in [19, 39] else 0.0
            info = {
                "action_type": "summarize",
                "budget_summary": dict(self.category_totals),
                "total_reward": reward,
            }
            self.rewards.append(reward)
            self.step_infos.append(info)
            obs = self._make_observation()
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        if action.action_type == ActionType.MARK_DUPLICATE:
            reward = 1.8 if current_txn.ground_truth_flag == FlagType.DUPLICATE else -0.5
            info = {
                "action_type": "mark_duplicate",
                "duplicate_of": action.duplicate_of_id,
                "correct": current_txn.ground_truth_flag == FlagType.DUPLICATE,
                "total_reward": reward,
            }
            self.flagged_count += 1
            self._update_history(current_txn, action)
            self.rewards.append(reward)
            self.step_infos.append(info)
            self.final_transactions.append(current_txn)
            self.final_actions.append(action)
            self.current_step += 1
            return self._check_done_or_next(reward, info)

        # ── CATEGORIZE — main path ────────────────────────────

        reward, info = compute_reward(
            action=action,
            transaction=current_txn,
            merchant_history=self.merchant_history,
            action_counts=self.action_type_counts,
            spending_velocity_24h=self._get_velocity_24h(current_txn.timestamp),
            episode_step=self.current_step,
        )

        # Update running state
        self._update_financials(current_txn, action)
        self._update_history(current_txn, action)
        self._update_merchant_state(current_txn, action)

        if action.flag and action.flag != FlagType.NORMAL:
            self.flagged_count += 1

        self.actions_taken.append(action)
        self.rewards.append(reward)
        self.step_infos.append(info)
        self.final_transactions.append(current_txn)
        self.final_actions.append(action)
        self.current_step += 1

        return self._check_done_or_next(reward, info)

    # ─────────────────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────────────────

    def state(self) -> Optional[Observation]:
        if self.done:
            return None
        return self._make_observation()

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _check_done_or_next(self, reward: float, info: dict) -> StepResult:
        """Check if episode is done, handle deferred transactions."""

        # If we've processed all main transactions, process deferred ones
        if self.current_step >= len(self.transactions):
            if self.deferred_transactions:
                # Inject deferred transactions back into the queue
                next_deferred = self.deferred_transactions.pop(0)
                self.transactions.append(next_deferred)
                obs = self._make_observation(is_deferred=True)
                return StepResult(observation=obs, reward=reward, done=False, info=info)
            else:
                # Truly done
                self.done = True
                bonus, bonus_info = self._compute_final_bonus()
                total_reward = round(reward + bonus, 2)
                info["episode_bonus"] = bonus_info
                info["episode_done"] = True
                return StepResult(observation=None, reward=total_reward, done=True, info=info)

        obs = self._make_observation()

        # If next transaction is from a disputed merchant, pre-flag it
        next_txn = self.transactions[self.current_step]
        if next_txn.upi_id in self.disputed_merchants:
            info["disputed_merchant_alert"] = (
                f"Next transaction from disputed merchant: {next_txn.merchant_name}"
            )

        return StepResult(observation=obs, reward=reward, done=False, info=info)

    def _make_observation(
        self,
        info_context: Optional[str] = None,
        is_deferred: bool = False
    ) -> Observation:
        txn = self.transactions[self.current_step]
        total = len(self.transactions)

        # Build spending trends
        trends = [
            SpendingTrend(
                category=cat,
                total_spent=self.category_totals.get(cat, 0.0),
                transaction_count=self.category_counts.get(cat, 0),
                average_amount=(
                    self.category_totals.get(cat, 0.0) /
                    self.category_counts.get(cat, 1)
                    if self.category_counts.get(cat, 0) > 0 else 0.0
                ),
            )
            for cat in self.category_counts
        ]

        # Build merchant patterns
        patterns = [
            MerchantPattern(
                merchant_name=self.transactions[self.current_step].merchant_name,
                upi_id=upi_id,
                count=self.merchant_counts.get(upi_id, 0),
                total_amount=sum(self.merchant_amounts.get(upi_id, [])),
                average_amount=(
                    sum(self.merchant_amounts.get(upi_id, [])) /
                    max(len(self.merchant_amounts.get(upi_id, [])), 1)
                ),
                agent_categories_used=[
                    c.value for c in self.merchant_history.get(upi_id, [])
                ],
            )
            for upi_id in list(self.merchant_counts.keys())[:10]
        ]

        # Time since last transaction
        time_since = None
        if self.last_timestamp:
            try:
                last_dt = datetime.fromisoformat(self.last_timestamp)
                curr_dt = datetime.fromisoformat(txn.timestamp)
                time_since = abs((curr_dt - last_dt).total_seconds())
            except Exception:
                pass

        return Observation(
            transaction_id=txn.id,
            amount=txn.amount,
            merchant_name=txn.merchant_name,
            upi_id=txn.upi_id,
            timestamp=txn.timestamp,
            transaction_type=txn.transaction_type,
            description=txn.description,
            bank_ref=txn.bank_ref,

            # Sequential intelligence
            recent_history=self.recent_history[-5:],
            spending_trends=trends,
            merchant_patterns=patterns,
            spending_velocity_24h=self._get_velocity_24h(txn.timestamp),
            time_since_last_txn=time_since,
            deferred_count=len(self.deferred_transactions),
            disputed_merchants=list(self.disputed_merchants),
            info_context=info_context,
            is_deferred=is_deferred,

            # Episode context
            step=self.current_step,
            total_steps=total,
            transactions_remaining=total - self.current_step,
            total_spent=round(self.total_spent, 2),
            total_received=round(self.total_received, 2),
            category_counts=self.category_counts.copy(),
            flagged_count=self.flagged_count,
            difficulty=self.difficulty,
        )

    def _update_financials(self, txn: UPITransaction, action: AgentAction):
        if txn.transaction_type == TransactionType.DEBIT:
            self.total_spent += txn.amount
            if action.category:
                cat = action.category.value
                self.category_totals[cat] = self.category_totals.get(cat, 0.0) + txn.amount
        else:
            self.total_received += txn.amount

        if action.category:
            cat = action.category.value
            self.category_counts[cat] = self.category_counts.get(cat, 0) + 1

        # Track for velocity
        self.spending_last_24h.append((txn.timestamp, txn.amount))
        self.last_timestamp = txn.timestamp

    def _update_history(self, txn: UPITransaction, action: AgentAction):
        summary = TransactionSummary(
            id=txn.id,
            amount=txn.amount,
            merchant_name=txn.merchant_name,
            timestamp=txn.timestamp,
            transaction_type=txn.transaction_type,
            agent_category=action.category,
            agent_flag=action.flag,
        )
        self.recent_history.append(summary)
        if len(self.recent_history) > 10:
            self.recent_history = self.recent_history[-10:]

    def _update_merchant_state(self, txn: UPITransaction, action: AgentAction):
        key = txn.upi_id
        if key not in self.merchant_history:
            self.merchant_history[key] = []
            self.merchant_amounts[key] = []
            self.merchant_counts[key] = 0

        if action.category:
            self.merchant_history[key].append(action.category)
        self.merchant_amounts[key].append(txn.amount)
        self.merchant_counts[key] += 1

    def _get_velocity_24h(self, current_timestamp: str) -> float:
        try:
            curr_dt = datetime.fromisoformat(current_timestamp)
            total = sum(
                amt for ts, amt in self.spending_last_24h
                if abs((curr_dt - datetime.fromisoformat(ts)).total_seconds()) <= 86400
            )
            return round(total, 2)
        except Exception:
            return 0.0

    def _is_cryptic(self, txn: UPITransaction) -> bool:
        name = txn.merchant_name
        return (
            any(c.isdigit() for c in name) or
            len(name) < 5 or
            "/" in name or
            "*" in name
        )

    def _compute_final_bonus(self) -> tuple[float, dict]:
        correct_cats = sum(1 for i in self.step_infos if i.get("correct_category"))
        fraud_caught = sum(
            1 for txn, info in zip(self.final_transactions, self.step_infos)
            if txn.ground_truth_flag == FlagType.SUSPICIOUS and info.get("correct_flag")
        )
        total_fraud = sum(
            1 for t in self.final_transactions
            if t.ground_truth_flag == FlagType.SUSPICIOUS
        )
        false_positives = sum(1 for i in self.step_infos if i.get("false_alarm"))

        # Ground truth category totals
        gt_trends = {}
        for txn in self.final_transactions:
            if txn.transaction_type == TransactionType.DEBIT:
                cat = txn.ground_truth_category.value
                gt_trends[cat] = gt_trends.get(cat, 0.0) + txn.amount

        return compute_episode_bonus(
            total_correct_categories=correct_cats,
            total_transactions=len(self.final_transactions),
            fraud_caught=fraud_caught,
            total_fraud=total_fraud,
            false_positives=false_positives,
            spending_trends=self.category_totals,
            ground_truth_trends=gt_trends,
            action_type_counts=self.action_type_counts,
        )

    def get_episode_summary(self) -> dict:
        if not self.done:
            raise ValueError("Episode not done yet.")
        return {
            "transactions": self.final_transactions,
            "actions": self.final_actions,
            "rewards": self.rewards,
            "step_infos": self.step_infos,
            "total_reward": round(sum(self.rewards), 2),
            "difficulty": self.difficulty.value,
            "action_type_counts": self.action_type_counts,
            "disputed_merchants": self.disputed_merchants,
        }
