"""
Self Repair module.
Analyzes error messages, runtime exceptions, and linter outputs to propose patches
and execute autonomous test-repair loops.
"""
import re
from typing import Any

from agent.llm import LLMClient
from agent.memory import PersistentMemory


class SelfRepairLoop:
    """
    Manages autonomous feedback loop to self-repair buggy code or failing tests
    up to a configurable maximum of iterations.
    """
    def __init__(self, llm: LLMClient | None = None, memory: PersistentMemory | None = None):
        self.llm = llm or LLMClient()
        self.memory = memory or PersistentMemory()

    def parse_stack_trace(self, error_logs: str) -> dict[str, Any]:
        """
        Parses runtime logs/stack traces to find files and line numbers causing failures.
        """
        # Search for typical python Traceback file patterns: File "filepath", line X
        pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)'
        matches = re.findall(pattern, error_logs)

        if matches:
            # Get the last/deepest file in the trace
            filepath, line_str = matches[-1]
            return {
                "file_path": filepath,
                "line_number": int(line_str),
                "error_type": "Runtime exception in stack trace"
            }

        # Or look for simpler AssertionError or division by zero etc
        if "ZeroDivisionError" in error_logs:
            return {"file_path": None, "line_number": None, "error_type": "ZeroDivisionError"}

        return {
            "file_path": None,
            "line_number": None,
            "error_type": "Unknown error pattern"
        }

    def run_repair_iteration(self, filepath: str, current_code: str, error_logs: str) -> str:
        """
        Interacts with the LLM and memory systems to generate a targeted patch to fix the error.
        """
        # First check memory for similar bug fixes
        similar_fixes = self.memory.find_similar_fixes(error_logs)
        reusable_strategy = ""
        if similar_fixes:
            reusable_strategy = f"\nFound a highly relevant historical fix in memory:\n{similar_fixes[0]['successful_patch']}\n"

        prompt = f"""
You are an autonomous self-repair engineer.
We have a file at path '{filepath}' that is currently failing.
Here is the current content of the file:
```python
{current_code}
```

Here are the error logs and stack traces:
```
{error_logs}
```
{reusable_strategy}
Analyze the cause, resolve logical edge cases or syntax issues, and output ONLY the complete fixed source code of the file. No extra text, no markdown block wrappers.
"""
        fixed_code = self.llm.generate(prompt)

        # Clean up code wrappers if model still returns markdown blocks
        if fixed_code.startswith("```"):
            lines = fixed_code.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fixed_code = "\n".join(lines)

        return fixed_code
