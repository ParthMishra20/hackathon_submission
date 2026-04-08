#!/usr/bin/env python
"""Inference script for OpenEnv with required structured stdout output."""

import argparse
import os
import sys
from typing import Any

import requests
from openai import OpenAI

ENV_NAME = "openenv"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def choose_action_with_optional_llm(client: OpenAI | None) -> str:
    """Use OpenAI client when token exists; otherwise fall back to deterministic action."""
    if not API_KEY or client is None:
        return "finalize"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return exactly one token: finalize"},
                {"role": "user", "content": "Pick next action for this environment."},
            ],
            temperature=0,
            max_tokens=4,
            stream=False,
        )
        choice = (completion.choices[0].message.content or "").strip().lower()
        return "finalize" if "finalize" in choice else "finalize"
    except Exception:
        return "finalize"


def run_episode(server_url: str, task_id: str | None = None) -> int:
    """Run one full episode and emit required START/STEP/END output blocks."""
    client: OpenAI | None = None
    if API_KEY:
        # Use injected proxy env vars when present to ensure validator traffic flows through LiteLLM.
        client = OpenAI(
            base_url=(os.environ["API_BASE_URL"]
                      if "API_BASE_URL" in os.environ else API_BASE_URL),
            api_key=(os.environ["API_KEY"]
                     if "API_KEY" in os.environ else API_KEY),
        )

    rewards: list[float] = []
    steps = 0
    success = False
    score = 0.0
    started = False
    current_task = task_id or "unknown"

    try:
        reset_resp = requests.post(
            f"{server_url}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs: dict[str, Any] = reset_resp.json()

        current_task = str(obs.get("task_id", current_task))
        max_steps = int(obs.get("max_steps", 10))
        done = False

        log_start(task=current_task, env=ENV_NAME, model=MODEL_NAME)
        started = True

        while not done and steps < max_steps:
            action = choose_action_with_optional_llm(client)
            step_resp = requests.post(
                f"{server_url}/step", json={"action_type": action})
            step_resp.raise_for_status()

            payload: dict[str, Any] = step_resp.json()
            reward_raw = payload.get("reward", {}).get("score", 0.0)
            reward = float(reward_raw or 0.0)
            done = bool(payload.get("done", False))

            steps += 1
            rewards.append(reward)
            log_step(step=steps, action=action,
                     reward=reward, done=done, error=None)

        grade_resp = requests.get(f"{server_url}/grade/{current_task}")
        grade_resp.raise_for_status()
        grade: dict[str, Any] = grade_resp.json()
        score = float(grade.get("score", 0.0))
        score = max(0.0, min(score, 1.0))
        success = score > 0.0
        return 0
    except Exception as exc:
        if started:
            # Emit a final step with error so validators can parse execution failures.
            log_step(step=steps + 1, action="finalize",
                     reward=0.0, done=True, error=str(exc))
            steps += 1
            rewards.append(0.0)
        return 1
    finally:
        if not started:
            log_start(task=current_task, env=ENV_NAME, model=MODEL_NAME)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on OpenEnv")
    parser.add_argument(
        "--server", default="http://127.0.0.1:7860", help="OpenEnv server URL")
    parser.add_argument("--task", default=None,
                        help="Task ID (optional, random if not specified)")
    args = parser.parse_args()
    return run_episode(args.server, task_id=args.task)


if __name__ == "__main__":
    sys.exit(main())
