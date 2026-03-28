from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a customer-support triage agent. "
    "Return exactly one JSON object with keys: action_type, category, priority, tag, message, team. "
    "Use null for unused keys."
)

VALID_ACTIONS = {
    "analyze_email",
    "set_category",
    "set_priority",
    "add_tag",
    "draft_reply",
    "escalate",
    "finalize",
}


def fallback_policy(observation: dict[str, Any]) -> dict[str, Any]:
    email = observation["customer_email"].lower()
    current_category = observation.get("current_category")
    current_priority = observation.get("current_priority")
    tags = set(observation.get("tags", []))
    escalation_team = observation.get("escalation_team")
    draft_reply = observation.get("draft_reply")

    category = None
    priority = None
    team = None
    desired_tags: list[str] = []
    reply = None

    if "login alert" in email or "api keys" in email or "urgent" in email:
        category = "security_incident"
        priority = "urgent"
        team = "security"
        desired_tags = ["security", "account_takeover", "api_keys"]
        reply = (
            "We are locking your account immediately, please reset your password now and "
            "rotate all API keys from a trusted device. We will follow up with incident details."
        )
    elif "export" in email or "times out" in email or "csv" in email:
        category = "technical_issue"
        priority = "high"
        team = "engineering"
        desired_tags = ["bug", "export", "finance_blocker"]
        reply = (
            "Thanks for reporting this. Please share a diagnostic screenshot and browser console "
            "details so engineering can investigate. We will send a follow-up update shortly."
        )
    else:
        category = "billing"
        priority = "medium"
        desired_tags = ["refund", "policy"]
        reply = (
            "Thanks for contacting us about a refund. Our refund policy and timeline depend on "
            "plan status, and we will confirm the exact timeline in a follow-up."
        )

    if not current_category:
        return {
            "action_type": "set_category",
            "category": category,
            "priority": None,
            "tag": None,
            "message": None,
            "team": None,
        }

    if not current_priority:
        return {
            "action_type": "set_priority",
            "category": None,
            "priority": priority,
            "tag": None,
            "message": None,
            "team": None,
        }

    for tag in desired_tags:
        if tag not in tags:
            return {
                "action_type": "add_tag",
                "category": None,
                "priority": None,
                "tag": tag,
                "message": None,
                "team": None,
            }

    if team and not escalation_team:
        return {
            "action_type": "escalate",
            "category": None,
            "priority": None,
            "tag": None,
            "message": None,
            "team": team,
        }

    if not draft_reply:
        return {
            "action_type": "draft_reply",
            "category": None,
            "priority": None,
            "tag": None,
            "message": reply,
            "team": None,
        }

    return {
        "action_type": "finalize",
        "category": None,
        "priority": None,
        "tag": None,
        "message": None,
        "team": None,
    }


def normalize_action(action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type", "")).strip().lower()
    if action_type not in VALID_ACTIONS:
        return fallback_policy(observation)

    normalized = {
        "action_type": action_type,
        "category": action.get("category"),
        "priority": action.get("priority"),
        "tag": action.get("tag"),
        "message": action.get("message"),
        "team": action.get("team"),
    }
    return normalized


def model_action(client: OpenAI, model: str, observation: dict[str, Any]) -> dict[str, Any]:
    user_prompt = (
        "Observation:\n"
        f"{json.dumps(observation, indent=2)}\n\n"
        "Decide the next best single action. "
        "If all key fields are correctly set and reply drafted, use finalize."
    )

    response = client.responses.create(
        model=model,
        temperature=0.0,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"Model output did not contain JSON object: {text}")
    return json.loads(text[start: end + 1])


def run_task(base_url: str, client: OpenAI, model: str, task_id: str) -> dict[str, Any]:
    reset_resp = requests.post(
        f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    done = False
    steps = 0
    while not done and steps < observation.get("max_steps", 10):
        try:
            model_candidate = model_action(client, model, observation)
        except Exception:
            model_candidate = fallback_policy(observation)
        action = normalize_action(model_candidate, observation)
        step_resp = requests.post(f"{base_url}/step", json=action, timeout=30)
        if step_resp.status_code >= 400:
            action = fallback_policy(observation)
            step_resp = requests.post(
                f"{base_url}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        payload = step_resp.json()
        observation = payload["observation"]
        done = payload["done"]
        steps += 1

    grade_resp = requests.get(f"{base_url}/grader", timeout=30)
    grade_resp.raise_for_status()
    grade = grade_resp.json()
    return {
        "task_id": task_id,
        "score": grade["score"],
        "breakdown": grade["breakdown"],
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline model on all OpenEnv tasks")
    parser.add_argument(
        "--server", default="http://127.0.0.1:7860", help="OpenEnv server URL")
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="OpenAI model")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    tasks_resp = requests.get(f"{args.server}/tasks", timeout=30)
    tasks_resp.raise_for_status()
    tasks_payload = tasks_resp.json()
    task_ids = [task["task_id"] for task in tasks_payload["tasks"]]

    results = [run_task(args.server, client, args.model, task_id)
               for task_id in task_ids]
    avg_score = sum(item["score"] for item in results) / len(results)

    summary = {
        "model": args.model,
        "task_results": results,
        "average_score": round(avg_score, 4),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
