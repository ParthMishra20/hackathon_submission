from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import Action, ActionType, EnvironmentState, Observation, Reward, StepResult
from app.tasks import TASK_MAP, TASKS, TaskSpec


MAX_STEPS = 10
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


@dataclass
class SessionState:
    task: TaskSpec | None = None
    done: bool = False
    step_count: int = 0
    total_reward: float = 0.0
    current_category: str | None = None
    current_priority: str | None = None
    tags: set[str] = field(default_factory=set)
    escalation_team: str | None = None
    draft_reply: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class SupportTriageEnv:
    """OpenEnv-compatible customer-support triage simulation."""

    def __init__(self) -> None:
        self._task_idx = -1
        self._session = SessionState()

    def reset(self, task_id: str | None = None) -> Observation:
        if task_id:
            if task_id not in TASK_MAP:
                raise ValueError(f"Unknown task_id: {task_id}")
            task = TASK_MAP[task_id]
        else:
            self._task_idx = (self._task_idx + 1) % len(TASKS)
            task = TASKS[self._task_idx]

        self._session = SessionState(task=task)
        return self._build_observation(last_feedback="New ticket loaded")

    def state(self) -> EnvironmentState:
        task_id = self._session.task.task_id if self._session.task else None
        return EnvironmentState(
            task_id=task_id,
            done=self._session.done,
            step_count=self._session.step_count,
            max_steps=MAX_STEPS,
            total_reward=round(self._session.total_reward, 4),
            current_category=self._session.current_category,
            current_priority=self._session.current_priority,
            tags=sorted(self._session.tags),
            escalation_team=self._session.escalation_team,
            draft_reply=self._session.draft_reply,
            history=self._session.history,
        )

    def step(self, action: Action) -> StepResult:
        if not self._session.task:
            raise RuntimeError("Environment must be reset before step")

        if self._session.done:
            obs = self._build_observation(
                last_feedback="Episode already complete")
            reward = Reward(score=0.0, reason="Action after completion")
            return StepResult(observation=obs, reward=reward, done=True, info={"error": "done"})

        self._session.step_count += 1
        reward_value, reason = self._apply_action(action)

        if self._session.step_count >= MAX_STEPS:
            self._session.done = True
            reward_value -= 0.10
            reason = f"{reason}; max steps reached"

        if action.action_type == ActionType.FINALIZE:
            self._session.done = True
            grader = self.grade_episode()
            bonus = (grader["score"] - 0.5) * 0.4
            reward_value += bonus
            reason = f"{reason}; finalize bonus from grader"

        reward_value = max(0.0, min(1.0, reward_value))
        self._session.total_reward = max(
            0.0, min(1.0, self._session.total_reward + reward_value)
        )
        self._session.history.append(
            {
                "step": self._session.step_count,
                "action": action.model_dump(),
                "reward": round(reward_value, 4),
                "reason": reason,
            }
        )

        obs = self._build_observation(last_feedback=reason)
        reward = Reward(score=round(reward_value, 4), reason=reason)
        info = {
            "task_id": self._session.task.task_id,
            "total_reward": round(self._session.total_reward, 4),
        }
        return StepResult(observation=obs, reward=reward, done=self._session.done, info=info)

    def _apply_action(self, action: Action) -> tuple[float, str]:
        task = self._session.task
        assert task is not None

        if action.action_type == ActionType.ANALYZE_EMAIL:
            return 0.03, "Email reviewed"

        if action.action_type == ActionType.SET_CATEGORY:
            if not action.category:
                return -0.08, "Missing category"
            self._session.current_category = action.category
            return self._score_match(action.category, task.expected_category, 0.18, -0.05, "Category updated")

        if action.action_type == ActionType.SET_PRIORITY:
            if not action.priority:
                return -0.08, "Missing priority"
            priority_str = action.priority.value
            self._session.current_priority = priority_str
            return self._score_match(priority_str, task.expected_priority, 0.18, -0.06, "Priority updated")

        if action.action_type == ActionType.ADD_TAG:
            if not action.tag:
                return -0.05, "Missing tag"
            clean_tag = action.tag.strip().lower().replace(" ", "_")
            if clean_tag in self._session.tags:
                return -0.03, "Duplicate tag"
            self._session.tags.add(clean_tag)
            if clean_tag in task.expected_tags:
                return 0.08, "Useful tag added"
            return -0.02, "Tag added but not relevant"

        if action.action_type == ActionType.ESCALATE:
            if not action.team:
                return -0.08, "Missing team"
            team = action.team.strip().lower()
            self._session.escalation_team = team
            if task.requires_escalation and team == task.expected_escalation_team:
                return 0.20, "Correct escalation"
            if task.requires_escalation and team != task.expected_escalation_team:
                return -0.12, "Escalated to wrong team"
            return -0.10, "Unnecessary escalation"

        if action.action_type == ActionType.DRAFT_REPLY:
            if not action.message:
                return -0.08, "Missing reply message"
            message = action.message.strip()
            self._session.draft_reply = message
            keyword_hits = sum(
                1 for word in task.required_reply_keywords if word.lower() in message.lower()
            )
            quality = keyword_hits / max(1, len(task.required_reply_keywords))
            if len(message) < 40:
                return -0.04, "Reply too short"
            return (round(0.24 * quality, 4), "Reply drafted")

        if action.action_type == ActionType.FINALIZE:
            return 0.02, "Case finalized"

        return -0.10, "Unknown action"

    @staticmethod
    def _score_match(value: str, target: str, success_reward: float, fail_penalty: float, msg: str) -> tuple[float, str]:
        if value.strip().lower() == target.strip().lower():
            return success_reward, msg
        return fail_penalty, f"{msg}; incorrect value"

    def _build_observation(self, last_feedback: str) -> Observation:
        task = self._session.task
        if not task:
            raise RuntimeError("No task is loaded")
        return Observation(
            task_id=task.task_id,
            task_title=task.title,
            customer_email=task.email,
            allowed_categories=[
                "billing",
                "technical_issue",
                "security_incident",
                "account_management",
                "feature_request",
            ],
            allowed_priorities=["low", "medium", "high", "urgent"],
            allowed_teams=["billing", "engineering",
                           "security", "support_ops"],
            current_category=self._session.current_category,
            current_priority=self._session.current_priority,
            tags=sorted(self._session.tags),
            escalation_team=self._session.escalation_team,
            draft_reply=self._session.draft_reply,
            step_count=self._session.step_count,
            max_steps=MAX_STEPS,
            done=self._session.done,
            last_feedback=last_feedback,
        )

    def grade_episode(self) -> dict:
        if not self._session.task:
            raise RuntimeError("No active task")

        task = self._session.task
        category_score = 1.0 if (self._session.current_category or "").lower(
        ) == task.expected_category else 0.0
        priority_score = 1.0 if (self._session.current_priority or "").lower(
        ) == task.expected_priority else 0.0

        expected_tags = task.expected_tags
        got_tags = self._session.tags
        if expected_tags:
            tags_score = len(expected_tags.intersection(
                got_tags)) / len(expected_tags)
        else:
            tags_score = 1.0

        if task.requires_escalation:
            escalation_score = 1.0 if (
                self._session.escalation_team == task.expected_escalation_team) else 0.0
        else:
            escalation_score = 1.0 if self._session.escalation_team is None else 0.0

        reply = (self._session.draft_reply or "").lower()
        if task.required_reply_keywords:
            reply_score = sum(
                1 for kw in task.required_reply_keywords if kw in reply) / len(task.required_reply_keywords)
        else:
            reply_score = 1.0

        breakdown = {
            "category": round(category_score, 4),
            "priority": round(priority_score, 4),
            "tags": round(tags_score, 4),
            "escalation": round(escalation_score, 4),
            "reply": round(reply_score, 4),
        }
        score = (
            0.24 * breakdown["category"]
            + 0.20 * breakdown["priority"]
            + 0.20 * breakdown["tags"]
            + 0.16 * breakdown["escalation"]
            + 0.20 * breakdown["reply"]
        )
        strict_score = max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, score))
        return {
            "task_id": task.task_id,
            "score": round(strict_score, 4),
            "breakdown": breakdown,
            "passed": strict_score >= 0.75,
        }

    @staticmethod
    def list_tasks() -> list[dict]:
        return [
            {
                "id": t.task_id,
                "task_id": t.task_id,
                "name": t.task_id,
                "title": t.title,
                "difficulty": t.difficulty,
                "objective": t.objective,
                "grader": f"/grade/{t.task_id}",
                "has_grader": True,
                "grader_enabled": True,
                "grader_endpoint": f"/grade/{t.task_id}",
                "grader_url": f"/grade/{t.task_id}",
                "grader_config": {
                    "endpoint": f"/grade/{t.task_id}",
                    "method": "GET",
                },
            }
            for t in TASKS
        ]
