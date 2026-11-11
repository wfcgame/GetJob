"""Held-out intent cases for future generalization checks; never auto-executed."""

from app.intent_evaluation import LabeledExample


HELD_OUT_EXAMPLES: tuple[LabeledExample, ...] = (
    LabeledExample(
        "heldout-task-01", "explicit_task_creation",
        "Please add a to-do: book a dentist appointment.", "create_task",
    ),
    LabeledExample(
        "heldout-missing-01", "missing_information",
        "Add a task, please.", "clarify",
    ),
    LabeledExample(
        "heldout-missing-02", "missing_information",
        "Create a task for that report from our last chat.", "clarify",
    ),
    LabeledExample(
        "heldout-missing-03", "missing_information",
        "Set up a task to", "clarify",
    ),
    LabeledExample(
        "heldout-assist-01", "direct_assistance",
        "Which of these should become tasks?", "clarify",
    ),
    LabeledExample(
        "heldout-negation-01", "negation",
        "No task please; read my notes instead.", "clarify",
    ),
    LabeledExample(
        "heldout-restrict-01", "negation",
        "Draft a task to email Jordan, but do not save it until I say so.",
        "create_task",
    ),
    LabeledExample(
        "heldout-indirect-01", "indirect_task_request",
        "Could you put renew the car registration on my list?", "create_task",
    ),
    LabeledExample(
        "heldout-indirect-02", "indirect_task_request",
        "Maybe I should review my budget sometime.", None,
        "A statement of possible future work does not authorize creation; an explicit policy is needed.",
    ),
    LabeledExample(
        "heldout-negation-02", "negation",
        "Don't make a task for calling Sam; just tell me what to say.", "clarify",
    ),
)


HELD_OUT_RATIONALE = {
    "heldout-task-01": "A new explicit task phrasing checks the positive branch.",
    "heldout-missing-01": "Creation intent is clear, but the task description is absent.",
    "heldout-missing-02": "The requested task depends on unavailable prior context.",
    "heldout-missing-03": "An unfinished request needs a question before a proposal.",
    "heldout-assist-01": "Asking for advice does not authorize task creation.",
    "heldout-negation-01": "Explicit negation overrides a nearby task keyword.",
    "heldout-restrict-01": "A specific proposal is allowed; saving still needs approval.",
    "heldout-indirect-01": "A concrete list request implies a task proposal.",
    "heldout-indirect-02": "An indirect statement exposes an unresolved product policy.",
    "heldout-negation-02": "Negation remains decisive with a concrete task name.",
}
