"""
Security Filtering and Safety module.
Validates inputs and code output to block secrets leaks, malicious prompts,
dangerous commands, or unsafe/exploitative operations.
"""
import re


class SecurityManager:
    """
    Scans code, commands, and prompts for security compliance.
    Ensures safe, production-grade autonomous operation.
    """
    def __init__(self):
        # Patterns matching dangerous commands
        self.dangerous_commands = [
            r"\brm\s+-rf\s+/",
            r"\bchmod\s+777\b",
            r"\bcurl\s+.*\bsh\b",
            r"\bwget\s+.*\bsh\b",
            r"\b:(){ :|:& };:", # Fork bomb
            r"\bcat\s+/etc/passwd\b",
            r"\bcat\s+/etc/shadow\b"
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
        Validates shell commands against dangerous configurations.
        Returns (is_safe, error_message).
        """
        for pattern in self.dangerous_commands:
            if re.search(pattern, command):
                return False, f"Dangerous command pattern detected: {pattern}"
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
                # Mask secret value (replace string value part cleanly)
                sanitized = re.sub(r"=\s*['\"][A-Za-z0-9_\-]+['\"]", "='[MASKED_SECRET]'", sanitized)
                removed.append(match)

        return sanitized, removed

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Runs rules on generated code to block unsafe Python or JS features (e.g. unsafe eval).
        """
        # Block arbitrary eval or exec with untrusted string inputs
        if "eval(" in code and not ("literal_eval" in code):
            return False, "Dangerous function 'eval' detected in generated code."
        if "exec(" in code:
            return False, "Dangerous function 'exec' detected in generated code."
        if "os.system(" in code or "subprocess.Popen(" in code:
            # We want to encourage using robust APIs rather than shell invocations in core generated logic
            return True, "Caution: Process spawned inside code output. Proceeding with caution."

        return True, "Safe code block."

    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Blocks malicious inputs or injection payloads targeting LLM decision engines.
        """
        injection_keywords = [
            "ignore previous instructions",
            "system override",
            "bypass safety",
            "unlimited access"
        ]
        for kw in injection_keywords:
            if kw in prompt.lower():
                return False, f"Malicious prompt injection attempt detected: {kw}"
        return True, "Safe prompt."
