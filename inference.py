#!/usr/bin/env python
"""Inference script demonstrating agent interaction with OpenEnv."""

import argparse
import json
import sys

import requests


def run_episode(server_url: str, task_id: str | None = None) -> dict:
    """Run one full episode on a task."""
    # Reset
    reset_resp = requests.post(
        f"{server_url}/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    task_id = obs["task_id"]
    done = False
    steps = 0
    total_reward = 0.0

    print(f"Task: {task_id}")
    print(f"Email: {obs['customer_email'][:100]}...")

    # Simple deterministic policy: finalize immediately
    while not done and steps < obs.get("max_steps", 10):
        step_resp = requests.post(
            f"{server_url}/step",
            json={"action_type": "finalize"},
        )
        step_resp.raise_for_status()
        payload = step_resp.json()
        obs = payload["observation"]
        reward = payload["reward"]["score"]
        done = payload["done"]
        total_reward += reward
        steps += 1

    # Grade
    grade_resp = requests.get(f"{server_url}/grader")
    grade_resp.raise_for_status()
    grade = grade_resp.json()

    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "grader_score": grade["score"],
        "grader_breakdown": grade["breakdown"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on OpenEnv")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:7860",
        help="OpenEnv server URL",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task ID (optional, random if not specified)",
    )
    args = parser.parse_args()

    try:
        result = run_episode(args.server, task_id=args.task)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
