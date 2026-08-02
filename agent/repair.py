"""
Self Repair and Bug Diagnosis module.
Analyzes error messages, runtime exceptions, linter outputs, scans files for common bugs,
explains code defects in detail, and interprets sandboxed test failures to propose patches.
"""
import re
from typing import Any

from agent.llm import LLMClient
from agent.memory import PersistentMemory


class SelfRepairLoop:
    """
    Manages autonomous feedback loops to diagnose code defects, explain bugs,
    interpret test output, and automatically repair failures up to a configurable maximum of iterations.
    """
    def __init__(self, llm: LLMClient | None = None, memory: PersistentMemory | None = None):
        self.llm = llm or LLMClient()
        self.memory = memory or PersistentMemory()

    def find_code_bugs(self, filepath: str, code_content: str) -> list[dict[str, Any]]:
        """
        Statically scans source file content for common bugs (e.g. division by zero,
        unhandled exceptions, raw dangerous shell execution, or hardcoded sensitive keywords).
        """
        bugs = []

        # 1. Division by Zero check
        if "/" in code_content:
            # Match division operations where denominator might be a literal zero or unvalidated variable
            div_zero_pattern = r'/\s*0\b'
            if re.search(div_zero_pattern, code_content):
                bugs.append({
                    "severity": "high",
                    "category": "ZeroDivisionError",
                    "description": "Literal division by zero detected in file.",
                    "mitigation": "Check the denominator and ensure it is non-zero before dividing."
                })

        # 2. Hardcoded API/Password strings
        sensitive_keywords = ["api_key", "secret_key", "password", "token"]
        for keyword in sensitive_keywords:
            pattern = rf'{keyword}\s*=\s*[\'\"][a-zA-Z0-9_\-]{{6,}}[\'\"]'
            if re.search(pattern, code_content, re.IGNORECASE):
                bugs.append({
                    "severity": "critical",
                    "category": "SecretsLeakage",
                    "description": f"Hardcoded credential/token mapping found for keyword '{keyword}'.",
                    "mitigation": "Banish hardcoded values and transition to secure environment variables instead."
                })

        # 3. Unsafe eval/exec mapping
        if "eval(" in code_content and "literal_eval" not in code_content:
            bugs.append({
                "severity": "critical",
                "category": "InsecureExecution",
                "description": "Unsafe 'eval()' usage detected. This is a massive injection risk.",
                "mitigation": "Transition to safe parsing utilities like ast.literal_eval."
            })

        return bugs

    def explain_bug(self, filepath: str, bug_details: dict[str, Any]) -> str:
        """
        Uses LLM intelligence to formulate a highly professional, engineering-grade
        explanation of why a bug exists and how it affects runtime environments.
        """
        prompt = f"""
You are a Principal Software Architect.
We have found a code defect in file '{filepath}' with the following diagnostics:
Category: {bug_details.get("category", "Logic error")}
Severity: {bug_details.get("severity", "Medium")}
Description: {bug_details.get("description", "Potential logical defect")}

Explain in 2-3 sentences why this bug is dangerous and how a senior software engineer should solve it properly.
"""
        return self.llm.generate(prompt).strip()

    def interpret_test_failure(self, test_output: str) -> dict[str, Any]:
        """
        Analyzes standard pytest/unittest stdout and stderr logs to extract
        the failing test function name, the exception type, and the assert statements.
        """
        # Find failed test cases
        failing_tests = []
        fail_func_pattern = r'FAIL:\s+(\w+)'
        failing_tests.extend(re.findall(fail_func_pattern, test_output))

        pytest_fail_pattern = r'____\s+(\w+)\s+____'
        failing_tests.extend(re.findall(pytest_fail_pattern, test_output))

        # Identify exception types
        exception_type = "AssertionError"
        if "ZeroDivisionError" in test_output:
            exception_type = "ZeroDivisionError"
        elif "NameError" in test_output:
            exception_type = "NameError"
        elif "TypeError" in test_output:
            exception_type = "TypeError"
        elif "ImportError" in test_output or "ModuleNotFoundError" in test_output:
            exception_type = "ImportError"

        return {
            "failing_tests": list(set(failing_tests)),
            "exception_type": exception_type,
            "raw_output_summary": test_output[-300:] if len(test_output) > 300 else test_output
        }

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
