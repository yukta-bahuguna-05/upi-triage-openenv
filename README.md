---
title: UPI Triage OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# UPI Triage OpenEnv 🇮🇳

An OpenEnv-compatible environment for **Indian UPI transaction triage**.

An AI agent processes a stream of UPI transactions one by one — categorizing spending, detecting fraud, catching duplicates, and building a financial summary.

**Motivated by a real Indian problem:** 500M+ UPI users generate billions of transactions monthly with no automatic categorization, fraud detection, or spending insights. This environment trains and evaluates AI agents to solve exactly that.

*Validated by Razorpay's Fix My Itch dataset — itch score 90.5/100 from 50,000 Indian users.*

---

## The Problem

Every Indian with a UPI app ends the month with cryptic transaction names like:
- `POS/TXN/884432`
- `9876543210@paytm`
- `IRCTC-REFUND` (which is actually a debit — fraud)

There is no automatic categorization, no fraud detection, no duplicate alerts. This environment trains an AI agent to solve this problem.

---

## Environment Overview

| Property | Value |
|---|---|
| Tasks | 3 (easy, medium, hard) |
| Observation | UPI transaction + episode context |
| Action | Category + Flag per transaction |
| Reward | Dense — at every step |
| Grader | Deterministic, 0.0–1.0 |
| API | FastAPI over HTTP |

---

## Observation Space

What the agent sees at each step:

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction ID |
| `amount` | float | Amount in INR (₹) |
| `merchant_name` | string | Merchant or recipient name |
| `upi_id` | string | UPI ID (e.g. swiggy@icici) |
| `timestamp` | string | ISO datetime |
| `transaction_type` | enum | debit or credit |
| `description` | string? | User note if any |
| `bank_ref` | string | Bank reference number |
| `step` | int | Current step (0-indexed) |
| `total_steps` | int | Total transactions in episode |
| `transactions_remaining` | int | How many left |
| `total_spent` | float | Running total spent so far |
| `total_received` | float | Running total received so far |
| `category_counts` | object | Count per category so far |
| `flagged_count` | int | Number flagged so far |
| `difficulty` | enum | easy / medium / hard |

---

## Action Space

What the agent submits for each transaction:

| Field | Type | Required | Description |
|---|---|---|---|
| `transaction_id` | string | ✅ | Must match current transaction |
| `category` | enum | ✅ | Spending category |
| `flag` | enum | ✅ | Normal / suspicious / duplicate / needs_review |
| `confidence` | float | ❌ | 0.0 to 1.0 |
| `reasoning` | string | ❌ | Agent's explanation |

**Valid categories:**
`food_delivery`, `groceries`, `transport`, `entertainment`, `utilities`, `rent`, `shopping`, `medical`, `education`, `investment`, `subscription`, `transfer`, `bill_split`, `salary`, `refund`, `suspicious`, `other`

**Valid flags:**
`normal`, `suspicious`, `duplicate`, `needs_review`

---

## Tasks

### Easy — Basic UPI Categorization
- 20 transactions
- Clear merchant names (Swiggy, Netflix, IRCTC, Zerodha)
- No fraud, no duplicates
- Agent needs to read merchant names and assign categories
- **Expected baseline score: 0.75**

### Medium — Mixed Transactions with Anomalies
- 40 transactions (includes all easy + 20 new)
- Mix of clear merchants and P2P transfers (person names, phone numbers)
- 2 duplicate transactions
- 2 suspicious transactions (prize scam, unknown midnight debit)
- **Expected baseline score: 0.55**

### Hard — Real-World Cryptic UPI Triage
- 60 transactions (includes all medium + 20 new)
- Cryptic merchant codes (`POS/TXN/884432`, `SWGY*ORDER`)
- Phone numbers as merchant names (`9876543210`)
- Multiple fraud patterns:
  - Fake IRCTC refund (is actually a debit)
  - ₹1 test charge followed by ₹24,999 fraud
  - KYC update scam
  - Slightly-off duplicate (₹15,000 vs ₹15,001)
- Forgotten subscriptions
- **Expected baseline score: 0.35**

---

## Reward Design

Dense rewards at every step — not just end of episode.

| Event | Reward |
|---|---|
| Correct category | +1.0 |
| Partially correct category | +0.5 |
| Wrong category | 0.0 |
| Correctly flagged fraud | +2.0 |
| Missed fraud | -1.0 |
| False alarm on normal transaction | -0.5 |
| Correctly caught duplicate | +1.5 |
| Missed duplicate | -0.5 |
| High confidence + correct | +0.1 bonus |
| High confidence + wrong | -0.1 penalty |

**End of episode bonuses:**

| Achievement | Bonus |
|---|---|
| Accuracy ≥ 90% | +5.0 |
| Accuracy ≥ 80% | +3.0 |
| Caught all fraud | +3.0 |
| Zero false positives | +2.0 |

---

## Grading Logic

Final deterministic score (0.0–1.0) after episode ends:

```
score = 0.60 × category_score
      + 0.25 × fraud_recall
      + 0.10 × duplicate_accuracy
      + 0.05 × (1 - false_positive_rate)
```

Category score uses partial credit — close categories score 0.5 instead of 0.

---

## Setup and Usage

### Local Setup

```bash
# Clone and install
git clone <your-repo-url>
cd openenv-upi-triage
pip install -r requirements.txt

# Start the server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run baseline inference
export API_BASE_URL=http://localhost:8000
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=your_key_here
python inference.py
```

### Docker

```bash
docker build -t upi-triage-env .
docker run -p 8000:8000 upi-triage-env
```

### API Usage

```python
import requests

# Start episode
obs = requests.post("http://localhost:8000/reset", json={"difficulty": "easy"}).json()

# Submit action
result = requests.post("http://localhost:8000/step", json={
    "action": {
        "transaction_id": obs["transaction_id"],
        "category": "food_delivery",
        "flag": "normal",
        "confidence": 0.95
    }
}).json()

# Get final score
grade = requests.post("http://localhost:8000/grade").json()
print(f"Score: {grade['score']}")
```

---

## Baseline Scores

Scores from GPT-4o-mini baseline agent:

| Task | Score | Category Acc | Fraud Recall |
|---|---|---|---|
| Easy | ~0.75 | ~0.85 | N/A |
| Medium | ~0.55 | ~0.72 | ~0.50 |
| Hard | ~0.35 | ~0.55 | ~0.40 |

---

## Project Structure

```
openenv-upi-triage/
├── env/
│   ├── models.py       # Typed Pydantic models
│   ├── upi_env.py      # Main environment (reset/step/state)
│   └── reward.py       # Dense reward function
├── tasks/
│   └── generator.py    # Transaction datasets
├── graders/
│   └── grader.py       # Deterministic scorer
├── api/
│   └── app.py          # FastAPI server
├── inference.py        # Baseline agent (OpenAI client)
├── openenv.yaml        # OpenEnv spec metadata
├── Dockerfile          # Container
└── README.md
```

---

## Why UPI Triage?

India processes 17+ billion UPI transactions per month. Every Indian user faces the same problem — their transaction history is a mess of cryptic codes, random names, and no categorization. This is a validated problem (Razorpay Fix My Itch itch score: 90.5) affecting 500M+ users. A well-trained AI agent solving this problem has immediate, deployable real-world value.
