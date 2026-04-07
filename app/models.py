from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ANALYZE_EMAIL = "analyze_email"
    SET_CATEGORY = "set_category"
    SET_PRIORITY = "set_priority"
    ADD_TAG = "add_tag"
    DRAFT_REPLY = "draft_reply"
    ESCALATE = "escalate"
    FINALIZE = "finalize"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Action(BaseModel):
    action_type: ActionType = Field(...,
                                    description="The type of action to execute")
    category: str | None = Field(
        default=None, description="Category label for SET_CATEGORY")
    priority: Priority | None = Field(
        default=None, description="Priority for SET_PRIORITY")
    tag: str | None = Field(
        default=None, description="Single tag value for ADD_TAG")
    message: str | None = Field(
        default=None, description="Reply content for DRAFT_REPLY")
    team: str | None = Field(
        default=None, description="Escalation team for ESCALATE")


class Observation(BaseModel):
    task_id: str
    task_title: str
    customer_email: str
    allowed_categories: list[str]
    allowed_priorities: list[str]
    allowed_teams: list[str]
    current_category: str | None
    current_priority: str | None
    tags: list[str]
    escalation_team: str | None
    draft_reply: str | None
    step_count: int
    max_steps: int
    done: bool
    last_feedback: str


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class EnvironmentState(BaseModel):
    task_id: str | None
    done: bool
    step_count: int
    max_steps: int
    total_reward: float
    current_category: str | None
    current_priority: str | None
    tags: list[str]
    escalation_team: str | None
    draft_reply: str | None
    history: list[dict[str, Any]]


class TaskSummary(BaseModel):
    task_id: str
    title: str
    difficulty: str
    objective: str
    grader: str = "/grader"
    has_grader: bool = True
    grader_endpoint: str = "/grader"
    grader_url: str = "/grader"
    grader_config: dict[str, str] = Field(
        default_factory=lambda: {"endpoint": "/grader", "method": "GET"}
    )


class GraderSpec(BaseModel):
    task_id: str
    endpoint: str
    method: str = "GET"


class TaskListResponse(BaseModel):
    tasks: list[TaskSummary]
    action_schema: dict[str, Any]
    graders: list[GraderSpec]


class ResetRequest(BaseModel):
    task_id: str | None = Field(
        default=None, description="Optional explicit task id")


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]
    passed: bool
