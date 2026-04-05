"""
inference.py — Baseline agent for the UPI Triage OpenEnv Environment.

Uses a smart rule-based agent with:
  - Comprehensive merchant matching (handles abbreviations like SWGY*, AMZN*, NF*)
  - Duplicate detection via transaction memory
  - Smart fraud detection (odd hours, suspicious patterns, KYC scams)
  - Context-aware P2P classification (rent vs transfer vs bill_split)

Required environment variables:
  API_BASE_URL   — The environment API endpoint
  MODEL_NAME     — The model name (informational)
  HF_TOKEN       — Hugging Face token
  OPENAI_API_KEY — OpenAI API key (if using LLM mode)

Usage:
  python inference.py
"""

import os
import json
import requests
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Config — read from environment variables ONLY (never hardcode keys)
# ─────────────────────────────────────────────────────────────

API_BASE_URL   = os.environ.get("API_BASE_URL", "https://yuktabahuguna-upi-triage-openenv.hf.space")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ─────────────────────────────────────────────────────────────
# MERCHANT CATEGORY RULES
# Each entry: ([keywords], category)
# Keywords are checked against merchant name + description (lowercased)
# Handles full names AND common UPI abbreviations
# ─────────────────────────────────────────────────────────────

MERCHANT_RULES = [
    # Food delivery — SWGY* is Swiggy's abbreviated UPI name
    (["swiggy", "swgy", "zomato", "zmt", "food delivery",
      "dunzo food", "blinkit food"], "food_delivery"),

    # Groceries — QR/KIRANA/STORE is a common pattern
    (["bigbasket", "big basket", "dmart", "kirana", "grofers",
      "blinkit", "zepto", "vegetables", "sabzi", "grocery",
      "supermarket", "reliance fresh", "more store"], "groceries"),

    # Transport — AUTO/PAY/OLA is Ola's autopay pattern
    (["uber", "ola", "irctc", "petrol", "fuel", "hp/petrol",
      "hp petrol", "iocl", "rapido", "metro", "bus",
      "train", "flight", "indigo", "spicejet", "auto/pay",
      "autoriksha", "parking"], "transport"),

    # Shopping — AMZN* is Amazon's abbreviated name, POS/TXN is card swipe
    (["amazon", "amzn", "flipkart", "myntra", "meesho", "nykaa",
      "snapdeal", "ajio", "tatacliq", "pos/txn", "pos txn",
      "retail", "store", "shop", "mall"], "shopping"),

    # Subscription — NF*NETFLIX, AUTOPAY/LINKEDIN patterns
    (["netflix", "nf*", "nf *", "spotify", "amazon prime",
      "hotstar", "jiocinema", "youtube premium", "linkedin",
      "autopay/linkedin", "google one", "google storage",
      "autopay/cure", "cure.fit", "gpay/sub", "disney",
      "zee5", "sonyliv", "apple music"], "subscription"),

    # Entertainment (non-subscription)
    (["bookmyshow", "pvr", "inox", "ticketnew", "district app",
      "paytm insider"], "entertainment"),

    # Utilities
    (["bescom", "electricity", "water bill", "gas", "bsnl",
      "airtel bill", "jio bill", "tata sky", "d2h", "dth",
      "mahanagar", "mseb", "tneb", "bbmp", "society"],
     "utilities"),

    # Medical
    (["apollo", "pharmacy", "medplus", "netmeds", "1mg",
      "doctor", "hospital", "clinic", "max healthcare",
      "fortis", "manipal", "diagnostic", "lab test"],
     "medical"),

    # Education
    (["coursera", "udemy", "unacademy", "byju", "vedantu",
      "whitehat", "college fee", "university", "tuition",
      "school fee", "exam fee", "coaching"], "education"),

    # Investment
    (["zerodha", "groww", "paytm money", "kuvera", "coin",
      "sip", "mutual fund", "nps", "ppf", "lumpsum",
      "smallcase", "scripbox"], "investment"),

    # Salary
    (["salary", "sal credit", "payroll", "stipend", "wages",
      "salary credit"], "salary"),

    # Refund
    (["refund", "cashback", "reversal", "return credit",
      "cancelled order", "order refund"], "refund"),
]

# ─────────────────────────────────────────────────────────────
# FRAUD SIGNALS — keywords that strongly indicate fraud
# ─────────────────────────────────────────────────────────────

FRAUD_KEYWORDS = [
    "unknown merchant",
    "prize",
    "winner",
    "lottery",
    "kyc update",
    "kyc verify",
    "kyc fee",
    "verify/card",
    "card verify",
    "online store xyz",
    "irctc-refund",      # Fake refund that is actually a debit
    "processing fee",
    "claim fee",
    "otp fee",
    "reward claim",
]

# ─────────────────────────────────────────────────────────────
# EPISODE MEMORY — reset at start of each task
# ─────────────────────────────────────────────────────────────

seen_transactions = []
upi_amount_history = defaultdict(list)
upi_category_history = defaultdict(list)


def reset_memory():
    """Reset all per-episode memory."""
    global seen_transactions, upi_amount_history, upi_category_history
    seen_transactions = []
    upi_amount_history = defaultdict(list)
    upi_category_history = defaultdict(list)


# ─────────────────────────────────────────────────────────────
# CATEGORY LOGIC
# ─────────────────────────────────────────────────────────────

def get_category(merchant: str, upi_id: str, amount: float,
                 txn_type: str, description: str) -> str:
    m = merchant.lower().strip()
    desc = (description or "").lower()
    combined = m + " " + desc

    # Check merchant rules (handles abbreviations too)
    for keywords, category in MERCHANT_RULES:
        if any(kw in combined for kw in keywords):
            return category

    # Phone number only — P2P transfer
    digits_only = merchant.replace(" ", "").replace("+91", "").replace("-", "")
    if digits_only.isdigit() and len(digits_only) >= 8:
        past_amounts = upi_amount_history.get(upi_id, [])
        if past_amounts and all(abs(a - amount) < 200 for a in past_amounts):
            return "rent"   # Same large amount every time = rent
        if amount >= 8000:
            return "transfer"
        return "transfer"

    # Person name — two or more alphabetic words
    words = merchant.split()
    looks_like_person = (
        len(words) >= 2 and
        all(w.replace(".", "").isalpha() for w in words) and
        len(merchant) < 40
    )

    if looks_like_person:
        # Use description for hints
        if any(w in desc for w in ["rent", "house", "room", "pg", "flat", "accommodation"]):
            return "rent"
        if any(w in desc for w in ["split", "dinner", "lunch", "movie", "trip", "party"]):
            return "bill_split"
        if any(w in desc for w in ["salary", "advance", "bonus", "stipend"]):
            return "salary"
        if any(w in desc for w in ["borrow", "lend", "return", "borrowed", "returned"]):
            return "transfer"

        # Amount pattern — same large amount repeatedly = rent
        past_amounts = upi_amount_history.get(upi_id, [])
        if amount >= 5000 and past_amounts:
            if all(abs(a - amount) < 300 for a in past_amounts):
                return "rent"

        # Large one-off to person = transfer
        if amount >= 5000:
            return "transfer"

        # Small to person = bill split
        return "bill_split"

    # Credit transactions
    if txn_type == "credit":
        if amount >= 30000:
            return "salary"
        return "transfer"

    return "other"


# ─────────────────────────────────────────────────────────────
# FLAG LOGIC
# ─────────────────────────────────────────────────────────────

def get_flag(merchant: str, upi_id: str, amount: float, timestamp: str) -> str:
    m = merchant.lower().strip()

    # ── Duplicate detection ───────────────────────────────────
    # Check if we've seen same or very similar transaction recently
    for past in seen_transactions:
        same_upi    = past["upi_id"] == upi_id
        same_merch  = past["merchant"].lower().strip() == m
        close_amt   = abs(past["amount"] - amount) < 2.0  # within ₹2

        try:
            curr_dt = datetime.fromisoformat(timestamp)
            past_dt = datetime.fromisoformat(past["timestamp"])
            secs    = abs((curr_dt - past_dt).total_seconds())
            close_time = secs < 300  # within 5 minutes
        except Exception:
            close_time = False

        if (same_upi or same_merch) and close_amt and close_time:
            return "duplicate"

    # ── Fraud keyword detection ───────────────────────────────
    for signal in FRAUD_KEYWORDS:
        if signal in m:
            return "suspicious"

    # ── Odd hour + unknown merchant ───────────────────────────
    try:
        hour = datetime.fromisoformat(timestamp).hour
        is_unknown = not any(
            kw in m
            for keywords, _ in MERCHANT_RULES
            for kw in keywords
        )
        if 0 <= hour < 5 and is_unknown and amount > 200:
            return "suspicious"
    except Exception:
        pass

    # ── Tiny test charge from unknown merchant ────────────────
    # Common pattern: ₹1-5 test charge followed by big fraud
    if amount <= 5.0:
        is_unknown = not any(
            kw in m
            for keywords, _ in MERCHANT_RULES
            for kw in keywords
        )
        if is_unknown:
            return "suspicious"

    # ── Salary should never be suspicious ────────────────────
    if "salary" in m or "sal credit" in m:
        return "normal"

    return "normal"


# ─────────────────────────────────────────────────────────────
# MAIN DECISION FUNCTION
# ─────────────────────────────────────────────────────────────

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

    # If duplicate detected, carry forward original category
    if flag == "duplicate":
        for past in reversed(seen_transactions):
            if past["upi_id"] == upi_id or \
               past["merchant"].lower() == merchant.lower():
                category = past["category"]
                break

    # Suspicious category always gets suspicious flag
    if category == "suspicious":
        flag = "suspicious"

    # Confidence based on how certain the rule match was
    clear_match = any(
        kw in merchant.lower()
        for keywords, _ in MERCHANT_RULES
        for kw in keywords
    )
    confidence = 0.92 if clear_match else 0.65

    # Store in memory
    seen_transactions.append({
        "id":        txn_id,
        "merchant":  merchant,
        "upi_id":    upi_id,
        "amount":    amount,
        "timestamp": timestamp,
        "category":  category,
        "flag":      flag,
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
# TASK RUNNER
# ─────────────────────────────────────────────────────────────

def run_task(difficulty: str) -> dict:
    print(f"\n{'='*58}")
    print(f"  Task: {difficulty.upper()}")
    print(f"{'='*58}")

    reset_memory()

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"difficulty": difficulty},
        headers=headers
    )
    response.raise_for_status()
    observation = response.json()

    step_count   = 0
    total_reward = 0.0

    while True:
        step_count += 1
        merchant = observation.get("merchant_name", "?")
        amount   = observation.get("amount", 0)

        action = categorize_transaction(observation)

        print(
            f"  {step_count:2d} | ₹{amount:>9.2f} | {merchant[:26]:<26}"
            f" → {action['category']:<18} [{action['flag']}]"
        )

        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"action": action},
            headers=headers
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        total_reward += result.get("reward", 0)

        if result.get("done"):
            break
        observation = result.get("observation")
        if observation is None:
            break

    grade_resp = requests.post(f"{API_BASE_URL}/grade", headers=headers)
    grade_resp.raise_for_status()
    grade_result = grade_resp.json()

    print(f"\n  ── {difficulty.upper()} Results ──")
    print(f"    Score               : {grade_result['score']:.4f}")
    print(f"    Category accuracy   : {grade_result['category_accuracy']:.4f}")
    print(f"    Fraud recall        : {grade_result['fraud_recall']:.4f}")
    print(f"    Duplicate accuracy  : {grade_result['duplicate_accuracy']:.4f}")
    print(f"    False positive rate : {grade_result['false_positive_rate']:.4f}")
    print(f"    Total reward        : {total_reward:.2f}")

    return grade_result


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("  UPI Triage OpenEnv — Baseline Inference")
    print("=" * 58)
    print(f"  API : {API_BASE_URL}")
    print(f"  Model : {MODEL_NAME}")

    try:
        h = requests.get(f"{API_BASE_URL}/health")
        h.raise_for_status()
        print(f"  Status : {h.json()['status'].upper()}")
    except Exception:
        print(f"  ERROR: Cannot connect to {API_BASE_URL}")
        print(f"  Run: uvicorn api.app:app --host 0.0.0.0 --port 8000")
        return

    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            results[difficulty] = run_task(difficulty)
        except Exception as e:
            print(f"\n  ERROR on {difficulty}: {e}")
            results[difficulty] = {"score": 0.0}

    print("\n" + "=" * 58)
    print("  FINAL SCORES")
    print("=" * 58)
    for d, r in results.items():
        print(f"  {d.upper():<10} : {r.get('score', 0):.4f}")
    avg = sum(r.get("score", 0) for r in results.values()) / len(results)
    print(f"  {'AVERAGE':<10} : {avg:.4f}")
    print("=" * 58)

    return results


if __name__ == "__main__":
    main()
