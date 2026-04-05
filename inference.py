"""
inference.py — Baseline agent for UPI Triage OpenEnv.

Follows the exact hackathon stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required environment variables:
  API_BASE_URL   — LLM API endpoint
  MODEL_NAME     — model identifier
  HF_TOKEN       — Hugging Face / API key
"""

import os
import json
import requests
from datetime import datetime
from collections import defaultdict
from typing import List, Optional
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL", "https://yuktabahuguna-upi-triage-openenv.hf.space")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BENCHMARK      = "upi-triage-openenv"

# OpenAI client for LLM calls
client = OpenAI(
    api_key=OPENAI_API_KEY or HF_TOKEN,
    base_url=None,
)

# ─────────────────────────────────────────────────────────────
# STDOUT LOGGING — exact required format
# ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# MERCHANT CATEGORY RULES
# ─────────────────────────────────────────────────────────────

MERCHANT_RULES = [
    (["swiggy", "swgy", "zomato", "zmt", "food delivery"], "food_delivery"),
    (["bigbasket", "big basket", "dmart", "kirana", "grofers",
      "blinkit", "zepto", "vegetables", "grocery"], "groceries"),
    (["uber", "ola", "irctc", "petrol", "fuel", "hp/petrol",
      "rapido", "metro", "train", "flight", "indigo", "auto/pay"], "transport"),
    (["amazon", "amzn", "flipkart", "myntra", "meesho", "nykaa",
      "pos/txn", "retail", "store", "mall"], "shopping"),
    (["netflix", "nf*", "spotify", "amazon prime", "hotstar",
      "linkedin", "autopay/linkedin", "google one", "autopay/cure",
      "cure.fit", "disney", "zee5"], "subscription"),
    (["bookmyshow", "pvr", "inox"], "entertainment"),
    (["bescom", "electricity", "water bill", "gas", "bsnl",
      "airtel bill", "jio bill", "tata sky", "d2h"], "utilities"),
    (["apollo", "pharmacy", "medplus", "netmeds", "1mg",
      "doctor", "hospital", "clinic", "max healthcare"], "medical"),
    (["coursera", "udemy", "unacademy", "byju", "vedantu",
      "college fee", "tuition", "coaching"], "education"),
    (["zerodha", "groww", "paytm money", "sip",
      "mutual fund", "lumpsum", "smallcase"], "investment"),
    (["salary", "sal credit", "payroll", "stipend"], "salary"),
    (["refund", "cashback", "reversal", "return credit"], "refund"),
]

FRAUD_KEYWORDS = [
    "unknown merchant", "prize", "winner", "lottery",
    "kyc update", "kyc verify", "kyc fee", "verify/card",
    "online store xyz", "irctc-refund", "processing fee",
    "claim fee", "otp fee",
]

# ─────────────────────────────────────────────────────────────
# EPISODE MEMORY
# ─────────────────────────────────────────────────────────────

seen_transactions   = []
upi_amount_history  = defaultdict(list)
upi_category_history = defaultdict(list)


def reset_memory():
    global seen_transactions, upi_amount_history, upi_category_history
    seen_transactions    = []
    upi_amount_history   = defaultdict(list)
    upi_category_history = defaultdict(list)


# ─────────────────────────────────────────────────────────────
# CATEGORY + FLAG LOGIC
# ─────────────────────────────────────────────────────────────

def get_category(merchant, upi_id, amount, txn_type, description):
    m        = merchant.lower().strip()
    desc     = (description or "").lower()
    combined = m + " " + desc

    for keywords, category in MERCHANT_RULES:
        if any(kw in combined for kw in keywords):
            return category

    digits_only = merchant.replace(" ", "").replace("+91", "").replace("-", "")
    if digits_only.isdigit() and len(digits_only) >= 8:
        past = upi_amount_history.get(upi_id, [])
        if past and all(abs(a - amount) < 200 for a in past):
            return "rent"
        return "transfer"

    words = merchant.split()
    looks_like_person = (
        len(words) >= 2 and
        all(w.replace(".", "").isalpha() for w in words) and
        len(merchant) < 40
    )

    if looks_like_person:
        if any(w in desc for w in ["rent", "house", "room", "pg", "flat"]):
            return "rent"
        if any(w in desc for w in ["split", "dinner", "lunch", "movie", "trip"]):
            return "bill_split"
        if any(w in desc for w in ["salary", "advance", "bonus"]):
            return "salary"
        past = upi_amount_history.get(upi_id, [])
        if amount >= 5000 and past and all(abs(a - amount) < 300 for a in past):
            return "rent"
        if amount >= 5000:
            return "transfer"
        return "bill_split"

    if txn_type == "credit":
        return "salary" if amount >= 30000 else "transfer"

    return "other"


def get_flag(merchant, upi_id, amount, timestamp):
    m = merchant.lower().strip()

    # Duplicate detection
    for past in seen_transactions:
        same_upi   = past["upi_id"] == upi_id
        same_merch = past["merchant"].lower().strip() == m
        close_amt  = abs(past["amount"] - amount) < 2.0
        try:
            curr_dt    = datetime.fromisoformat(timestamp)
            past_dt    = datetime.fromisoformat(past["timestamp"])
            close_time = abs((curr_dt - past_dt).total_seconds()) < 300
        except Exception:
            close_time = False
        if (same_upi or same_merch) and close_amt and close_time:
            return "duplicate"

    # Fraud keywords
    for signal in FRAUD_KEYWORDS:
        if signal in m:
            return "suspicious"

    # Odd hour + unknown
    try:
        hour       = datetime.fromisoformat(timestamp).hour
        is_unknown = not any(
            kw in m for keywords, _ in MERCHANT_RULES for kw in keywords
        )
        if 0 <= hour < 5 and is_unknown and amount > 200:
            return "suspicious"
    except Exception:
        pass

    # Tiny test charge from unknown
    if amount <= 5.0:
        is_unknown = not any(
            kw in m for keywords, _ in MERCHANT_RULES for kw in keywords
        )
        if is_unknown:
            return "suspicious"

    if "salary" in m:
        return "normal"

    return "normal"


def categorize_transaction(observation: dict) -> dict:
    txn_id    = observation["transaction_id"]
    merchant  = observation["merchant_name"]
    upi_id    = observation.get("upi_id", "")
    amount    = observation["amount"]
    timestamp = observation["timestamp"]
    txn_type  = observation.get("transaction_type", "debit")
    desc      = observation.get("description") or ""

    category = get_category(merchant, upi_id, amount, txn_type, desc)
    flag     = get_flag(merchant, upi_id, amount, timestamp)

    if flag == "duplicate":
        for past in reversed(seen_transactions):
            if past["upi_id"] == upi_id or past["merchant"].lower() == merchant.lower():
                category = past["category"]
                break

    if category == "suspicious":
        flag = "suspicious"

    clear_match = any(
        kw in merchant.lower()
        for keywords, _ in MERCHANT_RULES
        for kw in keywords
    )
    confidence = 0.92 if clear_match else 0.65

    seen_transactions.append({
        "id": txn_id, "merchant": merchant, "upi_id": upi_id,
        "amount": amount, "timestamp": timestamp,
        "category": category, "flag": flag,
    })
    upi_amount_history[upi_id].append(amount)
    upi_category_history[upi_id].append(category)

    return {
        "transaction_id": txn_id,
        "category":       category,
        "flag":           flag,
        "confidence":     confidence,
        "reasoning":      f"merchant='{merchant}' amount=₹{amount} type={txn_type}",
    }


# ─────────────────────────────────────────────────────────────
# TASK RUNNER — follows exact required stdout format
# ─────────────────────────────────────────────────────────────

def run_task(difficulty: str) -> dict:
    reset_memory()

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    # Reset environment
    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"difficulty": difficulty},
        headers=headers
    )
    response.raise_for_status()
    observation = response.json()

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    step_count   = 0
    rewards      = []
    score        = 0.0
    success      = False

    try:
        while True:
            step_count += 1
            action     = categorize_transaction(observation)
            action_str = f"categorize(id={action['transaction_id']},cat={action['category']},flag={action['flag']})"

            step_resp = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action},
                headers=headers
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            error  = result.get("info", {}).get("error", None)

            rewards.append(reward)
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            observation = result.get("observation")
            if observation is None:
                break

        # Get final grade
        grade_resp = requests.post(f"{API_BASE_URL}/grade", headers=headers)
        grade_resp.raise_for_status()
        grade_result = grade_resp.json()

        score   = grade_result.get("score", 0.0)
        success = score >= 0.5

    except Exception as e:
        log_end(success=False, steps=step_count, score=0.0, rewards=rewards)
        return {"score": 0.0, "error": str(e)}

    log_end(success=success, steps=step_count, score=score, rewards=rewards)
    return grade_result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Health check
    try:
        h = requests.get(f"{API_BASE_URL}/health")
        h.raise_for_status()
    except Exception:
        print(f"ERROR: Cannot connect to {API_BASE_URL}", flush=True)
        return

    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            results[difficulty] = run_task(difficulty)
        except Exception as e:
            results[difficulty] = {"score": 0.0}
            log_end(success=False, steps=0, score=0.0, rewards=[])

    # Summary
    print("\n" + "=" * 55, flush=True)
    print("  FINAL SCORES", flush=True)
    print("=" * 55, flush=True)
    for d, r in results.items():
        print(f"  {d.upper():<10} : {r.get('score', 0):.4f}", flush=True)
    avg = sum(r.get("score", 0) for r in results.values()) / len(results)
    print(f"  {'AVERAGE':<10} : {avg:.4f}", flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    main()
