from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException

from app.environment import SupportTriageEnv
from app.models import Action, GraderResponse, ResetRequest, StepResult, TaskListResponse, TaskSummary

app = FastAPI(title="OpenEnv Support Triage Environment", version="0.1.0")
ENV = SupportTriageEnv()
ROOT = Path(__file__).resolve().parent.parent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest | None = Body(default=None)):
    try:
        task_id = req.task_id if req else None
        return ENV.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        return ENV.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state():
    return ENV.state()


@app.get("/tasks", response_model=TaskListResponse)
def tasks():
    task_rows = [TaskSummary(**row) for row in ENV.list_tasks()]
    return TaskListResponse(
        tasks=task_rows,
        action_schema=Action.model_json_schema(),
        graders=[
            {"task_id": row.task_id, "endpoint": "/grader", "method": "GET"}
            for row in task_rows
        ],
    )


@app.get("/grader", response_model=GraderResponse)
def grader():
    try:
        result = ENV.grade_episode()
        return GraderResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/baseline")
def baseline():
    env = os.environ.copy()
    script_path = ROOT / "baseline" / "run_baseline.py"
    cmd = [sys.executable, str(script_path), "--server",
           "http://127.0.0.1:7860"]

    try:
        proc = subprocess.run(cmd, capture_output=True,
                              text=True, cwd=ROOT, env=env, check=True)
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        return parsed
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Baseline script failed",
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        ) from exc
    except (json.JSONDecodeError, IndexError) as exc:
        raise HTTPException(
            status_code=500, detail="Baseline output was not valid JSON") from exc
