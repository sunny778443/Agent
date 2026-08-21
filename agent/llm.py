"""
LLM interface module.
Handles integration with OpenAI and Anthropic API models.
Defaults to offline mock simulation mode unless LLM_PROVIDER and API keys are configured.
"""
import json
import logging
import os
from typing import Any

logger = logging.getLogger("agent.llm")


class LLMClient:
    """
    LLM client supporting Anthropic, OpenAI, or defaulting to offline mock simulation mode.
    Requires setting LLM_PROVIDER ('openai' or 'anthropic') and API key env vars for live API calls.
    """
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock")
        self.model = model or os.getenv("LLM_MODEL", "mock-model")

        self.mock_responses: list[str] = []
        self.history: list[dict[str, Any]] = []

        if self.provider == "mock":
            logger.info("LLMClient initialized in offline mock simulation mode.")

    def is_available(self) -> bool:
        """Returns True if live LLM provider or mock responses are configured."""
        if self.provider == "mock":
            return True
        return bool(
            (self.provider == "openai" and os.getenv("OPENAI_API_KEY"))
            or (self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
        )

    def set_mock_responses(self, responses: list[str]) -> None:
        """Sets simulated responses for mock mode."""
        self.mock_responses = responses

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        """
        Generates a completion based on provider and configurations.
        """
        self.history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "provider": self.provider,
            "model": self.model
        })

        if self.provider == "mock":
            if self.mock_responses:
                return self.mock_responses.pop(0)
            if "plan" in prompt.lower() or "planner" in prompt.lower():
                return json.dumps({
                    "steps": [
                        {"id": 1, "task": "Scan files and build AST", "risk": "low"},
                        {"id": 2, "task": "Write implementation", "risk": "medium"},
                        {"id": 3, "task": "Execute unit tests and static analysis", "risk": "low"}
                    ],
                    "risk_assessment": "Low risk with automated safety checks."
                })
            elif "repair" in prompt.lower() or "fix" in prompt.lower() or "error" in prompt.lower():
                return json.dumps({
                    "explanation": "Identified syntax/logic issue. Proposing fix.",
                    "code_patch": "def foo():\n    return 'fixed'"
                })
            elif "generate" in prompt.lower() or "code" in prompt.lower():
                return "def generated_code():\n    return 'production-grade'"
            return "Mock completion output."

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content or ""
            except (ImportError, RuntimeError, ValueError, AttributeError) as e:
                return f"Error executing OpenAI request: {e!s}"

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "mock-key"))
                system_arg = system_prompt if system_prompt else ""
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system=system_arg,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return "".join([block.text for block in response.content])
            except (ImportError, RuntimeError, ValueError, AttributeError) as e:
                return f"Error executing Anthropic request: {e!s}"

        return "Unsupported provider selected."
