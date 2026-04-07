from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException

from app.environment import SupportTriageEnv
from app.models import Action, EnvironmentState, GraderResponse, Observation, ResetRequest, StepResult, TaskListResponse, TaskSummary

try:
    from openenv.core.env_server.types import EnvironmentMetadata, HealthResponse, HealthStatus, SchemaResponse
except Exception:  # pragma: no cover
    EnvironmentMetadata = dict  # type: ignore[assignment]
    HealthResponse = dict  # type: ignore[assignment]
    HealthStatus = None  # type: ignore[assignment]
    SchemaResponse = dict  # type: ignore[assignment]

app = FastAPI(title="OpenEnv Support Triage Environment", version="0.1.0")
ENV = SupportTriageEnv()
ROOT = Path(__file__).resolve().parent.parent


@app.get("/metadata", response_model=EnvironmentMetadata)
def metadata():
    return {
        "name": "Support Triage OpenEnv",
        "description": "Real-world customer support triage simulation for training and evaluating agentic workflows.",
        "version": "0.1.0",
        "author": "OpenEnv Contributors",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


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


@app.get("/schema", response_model=SchemaResponse)
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.get("/tasks", response_model=TaskListResponse)
def tasks():
    task_rows = [TaskSummary(**row) for row in ENV.list_tasks()]
    return TaskListResponse(
        tasks=task_rows,
        action_schema=Action.model_json_schema(),
        graders=[
            {
                "id": row.task_id,
                "name": row.task_id,
                "task_id": row.task_id,
                "url": f"/grader/{row.task_id}",
                "endpoint": f"/grader/{row.task_id}",
                "method": "GET",
                "enabled": True,
            }
            for row in task_rows
        ],
    )


@app.get("/graders")
def graders():
    task_rows = [TaskSummary(**row) for row in ENV.list_tasks()]
    return {
        "graders": [
            {
                "id": row.task_id,
                "name": row.task_id,
                "task_id": row.task_id,
                "url": f"/grader/{row.task_id}",
                "endpoint": f"/grader/{row.task_id}",
                "method": "GET",
                "enabled": True,
            }
            for row in task_rows
        ]
    }


@app.get("/grader", response_model=GraderResponse)
def grader(task_id: str | None = None):
    try:
        if task_id:
            ENV.reset(task_id=task_id)
        result = ENV.grade_episode()
        return GraderResponse(**result)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/grader/{task_id}", response_model=GraderResponse)
def grader_for_task(task_id: str):
    try:
        ENV.reset(task_id=task_id)
        result = ENV.grade_episode()
        return GraderResponse(**result)
    except (RuntimeError, ValueError) as exc:
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
