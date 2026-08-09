"""
Plan Creator and Analyzer module.
Performs risk estimations, designs step-by-step executions, and sets priority tasks
with evidence-based confidence score calibration.
"""
import json
from typing import Any

from agent.llm import LLMClient


class Planner:
    """
    Formulates and analyzes step-by-step operations based on goals,
    estimating risk metrics and setting execution order with memory context enrichment.
    """
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def create_execution_plan(self, task_description: str) -> dict[str, Any]:
        """
        Creates a dynamic granular list of plan steps with associated risk scores
        prioritized by engineering impact using LLM intelligence.
        """
        return self.create_execution_plan_with_context(task_description, memory_context="")

    def create_execution_plan_with_context(self, task_description: str, memory_context: str = "", historical_success_rate: float | None = None, sample_count: int = 0) -> dict[str, Any]:
        """
        Generates an execution plan enriched with past memory and heuristics.
        Computes planner confidence scores dynamically based on actual historical success rates
        and retrieved experiences, or returns None if evidence is insufficient.
        """
        risk_score = 0.1
        if any(w in task_description.lower() for w in ["delete", "remove", "clean", "destroy"]):
            risk_score = 0.8
        elif any(w in task_description.lower() for w in ["refactor", "rewrite", "replace"]):
            risk_score = 0.5
        elif any(w in task_description.lower() for w in ["add", "new", "create", "implement"]):
            risk_score = 0.3

        # Compute evidence-based confidence metrics
        confidence_score = None
        observed_success_rate = None
        confidence_source = "None - Insufficient historical observations available to calibrate confidence stably."
        calculated_sample_count = sample_count

        if historical_success_rate is not None:
            confidence_score = round(historical_success_rate, 2)
            observed_success_rate = round(historical_success_rate, 2)
            confidence_source = "SQLite Experience Memory Logs"
        elif memory_context and "matching historical" in memory_context:
            try:
                # Parse matches count directly from context evidence string
                matches_count = int(memory_context.split("Found ")[1].split()[0])
                calculated_sample_count = matches_count
                if matches_count >= 1:
                    # Let's say we assume a conservative success expectation based purely on the sample density
                    # but we do NOT fabricate success probabilities out of thin air.
                    confidence_score = None
                    observed_success_rate = None
                    confidence_source = f"Uncalibrated - Found {matches_count} matching task files in memory"
            except Exception:
                pass

        if self.llm.provider == "mock":
            return {
                "task": task_description,
                "overall_risk_score": risk_score,
                "risk_assessment": "High risk" if risk_score > 0.6 else ("Medium risk" if risk_score > 0.3 else "Low risk"),
                "confidence_score": confidence_score,
                "sample_count": calculated_sample_count,
                "observed_success_rate": observed_success_rate,
                "confidence_source": confidence_source,
                "steps": [
                    {"id": 1, "action": "Analyze codebase architecture & file dependencies", "priority": "high", "verify": "Verify AST nodes parsed successfully"},
                    {"id": 2, "action": "Validate security boundaries and safety constraints", "priority": "high", "verify": "Confirm prompt does not violate guardrails"},
                    {"id": 3, "action": f"Develop implementation for: {task_description[:40]}", "priority": "medium", "verify": "Ensure no masked secrets are present"},
                    {"id": 4, "action": "Execute sandboxed verification and lint check", "priority": "high", "verify": "Ruff and MyPy return zero errors"}
                ]
            }

        prompt = f"""
Analyze this user task and generate a structured JSON execution plan enriched with the provided historical memory context:
Task: "{task_description}"
Historical Memory Context: "{memory_context}"

Provide response as a raw JSON dictionary with:
1. "overall_risk_score": (float between 0.0 and 1.0)
2. "risk_assessment": (string explanation of risks)
3. "steps": (list of dicts containing "id", "action", "priority" (high/medium/low), and "verify" (string self-verification check) keys)

Example output:
{{"overall_risk_score": 0.4, "risk_assessment": "Moderate Risk", "steps": [{{"id": 1, "action": "Implement logic", "priority": "high", "verify": "Check test success"}}]}}
"""
        raw_res = self.llm.generate(prompt)
        try:
            # strip markdown block ticks if returned
            if raw_res.strip().startswith("```"):
                cleaned = raw_res.strip().strip("`").strip("json").strip()
                data = json.loads(cleaned)
            else:
                data = json.loads(raw_res)

            data["confidence_score"] = confidence_score
            data["sample_count"] = calculated_sample_count
            data["observed_success_rate"] = observed_success_rate
            data["confidence_source"] = confidence_source
            return data
        except Exception:
            # Fallback
            return {
                "overall_risk_score": risk_score,
                "risk_assessment": "Low risk setup",
                "confidence_score": confidence_score,
                "sample_count": calculated_sample_count,
                "observed_success_rate": observed_success_rate,
                "confidence_source": confidence_source,
                "steps": [
                    {"id": 1, "action": f"Develop code satisfy: {task_description[:50]}", "priority": "high", "verify": "Build successfully"},
                    {"id": 2, "action": "Verify via sandbox", "priority": "high", "verify": "All unit tests pass"}
                ]
            }
