from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_openenv_yaml() -> None:
    path = ROOT / "openenv.yaml"
    if not path.exists():
        raise AssertionError("openenv.yaml is missing")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required_top_level = ["name", "description",
                          "entrypoint", "models", "additional_endpoints"]
    for key in required_top_level:
        if key not in data:
            raise AssertionError(f"openenv.yaml missing key: {key}")

    endpoints = data.get("entrypoint", {}).get("endpoints", {})
    for key in ["reset", "step", "state"]:
        if key not in endpoints:
            raise AssertionError(
                f"openenv.yaml entrypoint.endpoints missing: {key}")


def check_http_endpoints() -> None:
    from app.api import app

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200, "/health failed"

    tasks_resp = client.get("/tasks")
    assert tasks_resp.status_code == 200, "/tasks failed"
    tasks_payload = tasks_resp.json()

    tasks = tasks_payload.get("tasks", [])
    assert len(tasks) >= 3, "Need at least 3 tasks"

    difficulties = {t["difficulty"] for t in tasks}
    expected_difficulties = {"easy", "medium", "hard"}
    if not expected_difficulties.issubset(difficulties):
        raise AssertionError(
            "Task set must include easy, medium, and hard difficulties")

    action_schema = tasks_payload.get("action_schema", {})
    schema_props = action_schema.get("properties", {})
    if "action_type" not in schema_props:
        raise AssertionError("Action schema missing action_type")

    for task in tasks:
        reset_resp = client.post("/reset", json={"task_id": task["task_id"]})
        assert reset_resp.status_code == 200, f"/reset failed for {task['task_id']}"

        obs = reset_resp.json()
        assert "customer_email" in obs, "Observation missing customer_email"

        step_resp = client.post("/step", json={"action_type": "finalize"})
        assert step_resp.status_code == 200, f"/step failed for {task['task_id']}"

        state_resp = client.get("/state")
        assert state_resp.status_code == 200, "/state failed"
        assert state_resp.json().get("done") is True, "Episode should be done after finalize"

        grade_resp = client.get("/grader")
        assert grade_resp.status_code == 200, "/grader failed"
        score = grade_resp.json().get("score")
        assert isinstance(score, (int, float)), "Grader score must be numeric"
        assert 0.0 <= float(score) <= 1.0, "Grader score must be in [0.0, 1.0]"


def maybe_check_baseline_script() -> None:
    script = ROOT / "baseline" / "run_baseline.py"
    if not script.exists():
        raise AssertionError(
            "Baseline script missing: baseline/run_baseline.py")

    # Syntax check always runs, even without API key.
    compile(str(script.read_text(encoding="utf-8")), str(script), "exec")

    if not os.getenv("OPENAI_API_KEY"):
        print("WARN: OPENAI_API_KEY not set; skipping live baseline execution")
        return

    server_proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Give local server time to boot.
        import time

        time.sleep(2.0)
        cmd = [
            sys.executable,
            str(script),
            "--server",
            "http://127.0.0.1:7860",
            "--model",
            os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        ]
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        output = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(output)

        if "average_score" not in payload:
            raise AssertionError("Baseline output missing average_score")
        if "task_results" not in payload or len(payload["task_results"]) < 3:
            raise AssertionError(
                "Baseline output must include results for 3+ tasks")
    finally:
        server_proc.terminate()


def check_dockerfile_exists() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        raise AssertionError("Dockerfile is missing")


def main() -> None:
    print("[1/4] Checking openenv.yaml")
    check_openenv_yaml()

    print("[2/4] Checking API endpoints and graders")
    check_http_endpoints()

    print("[3/4] Checking baseline script")
    maybe_check_baseline_script()

    print("[4/4] Checking Dockerfile")
    check_dockerfile_exists()

    print("PASS: Pre-submission validation completed")


if __name__ == "__main__":
    main()
