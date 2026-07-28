"""
LLM interface module.
Handles integration with OpenAI and Anthropic API models.
Includes mock model capabilities for offline, sandbox, or testing operations.
"""
import json
import os
from typing import Any


class LLMClient:
    """
    A robust LLM client capable of communicating with Anthropic/OpenAI or falling back to custom mocks.
    """
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock")
        self.model = model or os.getenv("LLM_MODEL", "mock-model")

        # Simple local state to allow test mocking or manual overrides
        self.mock_responses: list[str] = []
        self.history: list[dict[str, Any]] = []

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
            # Default auto-responses based on prompt content
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
            except Exception as e:
                # Log or handle fallback
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
                # Anthropic message content is a list of TextBlock objects
                return "".join([block.text for block in response.content])
            except Exception as e:
                return f"Error executing Anthropic request: {e!s}"

        return "Unsupported provider selected."
