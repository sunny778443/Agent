"""
Plan Creator and Analyzer module.
Performs risk estimations, designs step-by-step executions, and sets priority tasks.
"""
import json
from typing import Any

from agent.llm import LLMClient


class Planner:
    """
    Formulates and analyzes step-by-step operations based on goals,
    estimating risk metrics and setting execution order.
    """
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def create_execution_plan(self, task_description: str) -> dict[str, Any]:
        """
        Creates a dynamic granular list of plan steps with associated risk scores
        prioritized by engineering impact using LLM intelligence.
        """
        if self.llm.provider == "mock":
            # Direct calculation for tests or mock mode
            risk_score = 0.1
            if any(w in task_description.lower() for w in ["delete", "remove", "clean", "destroy"]):
                risk_score = 0.8
            elif any(w in task_description.lower() for w in ["refactor", "rewrite", "replace"]):
                risk_score = 0.5
            elif any(w in task_description.lower() for w in ["add", "new", "create", "implement"]):
                risk_score = 0.3

            return {
                "task": task_description,
                "overall_risk_score": risk_score,
                "risk_assessment": "High risk" if risk_score > 0.6 else ("Medium risk" if risk_score > 0.3 else "Low risk"),
                "steps": [
                    {"id": 1, "action": "Analyze codebase architecture & file dependencies", "priority": "high"},
                    {"id": 2, "action": "Validate security boundaries and safety constraints", "priority": "high"}
                ]
            }

        prompt = f"""
Analyze this user task and generate a structured JSON execution plan:
"{task_description}"

Provide response as a raw JSON dictionary with:
1. "overall_risk_score": (float between 0.0 and 1.0)
2. "risk_assessment": (string explanation of risks)
3. "steps": (list of dicts containing "id", "action", "priority" keys)

Example output:
{{"overall_risk_score": 0.4, "risk_assessment": "Moderate Risk", "steps": [{{"id": 1, "action": "Implement logic", "priority": "high"}}]}}
"""
        raw_res = self.llm.generate(prompt)
        try:
            # strip markdown block ticks if returned
            if raw_res.strip().startswith("```"):
                cleaned = raw_res.strip().strip("`").strip("json").strip()
                return json.loads(cleaned)
            return json.loads(raw_res)
        except Exception:
            # Fallback
            return {
                "overall_risk_score": 0.3,
                "risk_assessment": "Low risk setup",
                "steps": [
                    {"id": 1, "action": f"Develop code satisfy: {task_description[:50]}", "priority": "high"},
                    {"id": 2, "action": "Verify via sandbox", "priority": "high"}
                ]
            }
