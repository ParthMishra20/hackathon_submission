from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str
    objective: str
    email: str
    expected_category: str
    expected_priority: str
    expected_tags: set[str]
    expected_escalation_team: str | None
    requires_escalation: bool
    required_reply_keywords: list[str]


TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="easy_refund_policy",
        title="Handle Basic Refund Policy Question",
        difficulty="easy",
        objective=(
            "Categorize a straightforward refund-policy inquiry, set medium priority, "
            "tag correctly, and draft a concise policy reply."
        ),
        email=(
            "Hi support, I bought your annual analytics plan yesterday by mistake and "
            "want to know if I can get a refund. I have not used the product yet. "
            "Thanks, Anika"
        ),
        expected_category="billing",
        expected_priority="medium",
        expected_tags={"refund", "policy"},
        expected_escalation_team=None,
        requires_escalation=False,
        required_reply_keywords=["refund", "policy", "timeline"],
    ),
    TaskSpec(
        task_id="medium_data_export_bug",
        title="Triage Data Export Failure",
        difficulty="medium",
        objective=(
            "Identify a product bug report, set high priority, include technical tags, "
            "and produce a reply requesting diagnostics while promising a follow-up."
        ),
        email=(
            "Hello team, every time I try to export invoices to CSV from Chrome on macOS, "
            "the app spins forever and then times out. This is blocking finance close. "
            "Can you help quickly? - Rafael"
        ),
        expected_category="technical_issue",
        expected_priority="high",
        expected_tags={"bug", "export", "finance_blocker"},
        expected_escalation_team="engineering",
        requires_escalation=True,
        required_reply_keywords=["diagnostic", "screenshot", "follow-up"],
    ),
    TaskSpec(
        task_id="hard_security_incident",
        title="Respond to Suspected Account Takeover",
        difficulty="hard",
        objective=(
            "Correctly identify a security incident, mark urgent, escalate to security, "
            "apply incident tags, and draft a safety-first response with immediate steps."
        ),
        email=(
            "URGENT: I received a login alert from a country I do not recognize, and now "
            "my API keys seem to have been regenerated. I did not do this. Please lock my "
            "account immediately and advise what to do next. - Mei"
        ),
        expected_category="security_incident",
        expected_priority="urgent",
        expected_tags={"security", "account_takeover", "api_keys"},
        expected_escalation_team="security",
        requires_escalation=True,
        required_reply_keywords=["lock", "password", "rotate"],
    ),
]


TASK_MAP = {task.task_id: task for task in TASKS}
