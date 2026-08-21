"""
Security Filtering and Pattern Analysis module.
Performs AST-based inspection and command pattern filtering on code, commands, and prompts
to detect unsafe functions, forbidden shell calls, and secret leaks.
Note: This module provides static rule and AST-based pattern analysis as a first-line filter.
It is not an absolute isolation boundary; code execution security is enforced by sandboxed execution environments.
"""

import ast
import re
import shlex


class SecurityManager:
    """
    Scans code, commands, and prompts using AST inspection and pattern matching
    to filter unsafe operations, secret leaks, and command injection patterns.
    """
    def __init__(self):
        # Forbidden executable names or dangerous command configurations
        self.forbidden_executables = {"rm", "chmod", "chown", "dd", "mkfs", "shutdown", "reboot"}

        # Patterns matching dangerous raw command strings
        self.dangerous_command_patterns = [
            r"\brm\s+-[a-zA-Z]*rf?\s+(/|\*|/\*)",
            r"\bchmod\s+777\b",
            r"\bcurl\s+.*\bsh\b",
            r"\bwget\s+.*\bsh\b",
            r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
            r"/etc/(passwd|shadow|sudoers)"
        ]

        # Patterns matching potentially leaked secrets (API keys, credentials, etc.)
        self.sensitive_patterns = [
            r"(?i)api[-_]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
            r"(?i)secret[-_]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
            r"(?i)password\s*=\s*['\"][A-Za-z0-9_\-]{6,}['\"]",
            r"(?i)token\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
            r"xox[bapr]-\d+-\d+-\w+",  # Slack tokens
            r"AIzaSy[A-Za-z0-9_\-]{33}"  # Google API Keys
        ]

    def validate_command(self, command: str) -> tuple[bool, str]:
        """
        Validates shell commands using shlex tokenization and regex patterns.
        Returns (is_safe, error_message).
        """
        if not command or not command.strip():
            return True, "Safe command."

        for pattern in self.dangerous_command_patterns:
            if re.search(pattern, command):
                return False, f"Dangerous command pattern detected: '{pattern}'"

        try:
            tokens = shlex.split(command)
            if tokens:
                binary = tokens[0].split("/")[-1]
                if binary in self.forbidden_executables and binary == "rm" and any(arg in tokens for arg in ["-rf", "-fr", "-r", "-f"]):
                    return False, f"Forbidden command executable and destructive flag detected: '{command}'"
        except ValueError:
            pass

        return True, "Safe command."

    def sanitize_code(self, code: str) -> tuple[str, list[str]]:
        """
        Removes/masks secrets or sensitive keys found in generated code files.
        Returns (sanitized_code, list_of_removed_secrets).
        """
        sanitized = code
        removed = []

        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, sanitized)
            for match in matches:
                sanitized = re.sub(r"=\s*['\"][A-Za-z0-9_\-]+['\"]", "='[MASKED_SECRET]'", sanitized)
                removed.append(match)

        return sanitized, removed

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Performs Python AST inspection to block unsafe Python functions
        (e.g., eval, exec, compile, os.system, dangerous attribute access).
        """
        if not code or not code.strip():
            return True, "Safe code block."

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error in code block: {e.msg} at line {e.lineno}"

        forbidden_functions = {"eval", "exec", "compile", "__import__"}
        forbidden_attributes = {"__subclasses__", "__globals__", "__mro__", "__bases__", "__builtins__"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_functions:
                    return False, f"Unsafe function call '{node.func.id}()' detected by AST inspection."

                if isinstance(node.func, ast.Attribute) and node.func.attr in ("system", "popen") and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    return False, f"Unsafe system invocation 'os.{node.func.attr}()' detected by AST inspection."

                if isinstance(node.func, ast.Attribute) and node.func.attr in ("Popen", "call", "run", "check_output"):
                    for keyword in node.keywords:
                        if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            return False, f"Unsafe subprocess call with 'shell=True' detected in '{node.func.attr}()'."

            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                return False, f"Unsafe magic attribute access '{node.attr}' detected by AST inspection."

        return True, "Safe code block."

    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Filters prompt injection patterns targeting decision engines.
        """
        if not prompt or not prompt.strip():
            return True, "Safe prompt."

        injection_keywords = [
            "ignore previous instructions",
            "system override",
            "bypass safety",
            "unlimited access"
        ]
        for kw in injection_keywords:
            if kw in prompt.lower():
                return False, f"Malicious prompt injection pattern detected: '{kw}'"
        return True, "Safe prompt."
