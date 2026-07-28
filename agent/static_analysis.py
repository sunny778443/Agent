"""
Static Analysis and Linting module.
Executes Ruff, MyPy, Pylint, and ESLint on targets, then automatically reads logs
and uses LLM or AST fixes to repair issues.
"""
import subprocess
from typing import Any

from agent.security import SecurityManager


class StaticAnalyzer:
    """
    Scans files with MyPy, Ruff, etc. parses outputs, and can execute clean up actions.
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.security = SecurityManager()

    def run_ruff(self, filepath: str) -> dict[str, Any]:
        """Runs Ruff linter on a specific file."""
        is_safe, msg = self.security.validate_command(f"ruff check {filepath}")
        if not is_safe:
            return {"success": False, "errors": [msg]}

        try:
            res = subprocess.run(
                ["ruff", "check", filepath],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            return {
                "success": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "errors": [] if res.returncode == 0 else [res.stdout + res.stderr]
            }
        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    def run_mypy(self, filepath: str) -> dict[str, Any]:
        """Runs MyPy type-checker on a file."""
        is_safe, msg = self.security.validate_command(f"mypy {filepath}")
        if not is_safe:
            return {"success": False, "errors": [msg]}

        try:
            res = subprocess.run(
                ["mypy", filepath, "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            return {
                "success": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "errors": [] if res.returncode == 0 else [res.stdout + res.stderr]
            }
        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    def repair_lint_issues(self, filepath: str, linter: str = "ruff") -> dict[str, Any]:
        """
        Attempts automatic command-line style repair (e.g., ruff format or ruff check --fix).
        """
        if linter == "ruff":
            try:
                # Format
                subprocess.run(["ruff", "format", filepath], cwd=self.root_dir)
                # Fix lints
                res = subprocess.run(["ruff", "check", "--fix", filepath], capture_output=True, text=True, cwd=self.root_dir)
                return {
                    "success": res.returncode == 0,
                    "stdout": res.stdout,
                    "stderr": res.stderr
                }
            except Exception as e:
                return {"success": False, "stderr": str(e)}
        return {"success": False, "stderr": f"Unsupported auto-repair linter: {linter}"}
