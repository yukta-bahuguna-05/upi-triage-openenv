"""
app.py — FastAPI server for the UPI Triage OpenEnv Environment.

Exposes standard OpenEnv HTTP endpoints:
  POST /reset          → start new episode
  POST /step           → submit action, get next state
  GET  /state          → get current state
  GET  /tasks          → list available tasks
  POST /grade          → get final score
  GET  /health         → health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback

from env.upi_env import UPITriageEnv
from env.models import AgentAction, Observation, StepResult, GradeResult
from graders.grader import grade


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="UPI Triage OpenEnv",
    description=(
        "An OpenEnv-compatible environment for UPI transaction triage. "
        "An AI agent processes Indian UPI transactions, categorizes them, "
        "detects fraud and duplicates, and builds a financial summary."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (stateful)
# In production you'd use sessions, but this is sufficient for the hackathon
env = UPITriageEnv(difficulty="easy")


# ─────────────────────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"  # easy / medium / hard


class StepRequest(BaseModel):
    action: AgentAction


class GradeRequest(BaseModel):
    difficulty: Optional[str] = None  # Override difficulty for grading


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — confirms server is running."""
    return {"status": "ok", "environment": "upi-triage", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Basic UPI Categorization",
                "description": (
                    "20 transactions with clear merchant names. "
                    "Agent must assign correct spending categories. "
                    "No fraud or duplicates."
                ),
                "num_transactions": 20,
                "has_fraud": False,
                "has_duplicates": False,
                "expected_baseline_score": 0.75,
            },
            {
                "id": "medium",
                "name": "Mixed Transactions with Anomalies",
                "description": (
                    "40 transactions including P2P transfers, "
                    "2 duplicate transactions, and 2 suspicious transactions. "
                    "Agent must categorize AND detect anomalies."
                ),
                "num_transactions": 40,
                "has_fraud": True,
                "has_duplicates": True,
                "expected_baseline_score": 0.55,
            },
            {
                "id": "hard",
                "name": "Real-World Cryptic UPI Triage",
                "description": (
                    "60 transactions with cryptic merchant IDs, "
                    "phone numbers as merchant names, multiple fraud patterns "
                    "(test charge + large fraud, KYC scam, phishing), "
                    "and hidden recurring patterns. "
                    "Requires deep reasoning and pattern recognition."
                ),
                "num_transactions": 60,
                "has_fraud": True,
                "has_duplicates": True,
                "expected_baseline_score": 0.35,
            },
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = None):
    global env
    try:
        difficulty = "easy"
        if request and request.difficulty:
            difficulty = request.difficulty
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "easy"
        env = UPITriageEnv(difficulty=difficulty)
        observation = env.reset()
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    """
    Submit agent action on current transaction.
    Returns next observation, reward, done status, and info.
    """
    try:
        result = env.step(request.action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state", response_model=Optional[Observation])
def state():
    """
    Get current observation without advancing the episode.
    Returns null if episode is done.
    """
    return env.state()


@app.post("/grade", response_model=GradeResult)
def grade_episode():
    """
    Grade the completed episode.
    Returns deterministic score 0.0 to 1.0 with full breakdown.
    Must be called after episode is done (all transactions processed).
    """
    try:
        summary = env.get_episode_summary()
        result = grade(
            transactions=summary["transactions"],
            actions=summary["actions"],
            difficulty=summary["difficulty"],
            action_type_counts=summary.get("action_type_counts", {}),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/")
def root():
    """Root endpoint — environment info."""
    return {
        "name": "UPI Triage OpenEnv",
        "description": "OpenEnv environment for Indian UPI transaction triage",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
        "tasks": ["easy", "medium", "hard"],
        "spec": "openenv-v1",
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()