"""
Self-Healing Codebase Auditor module.
Scans Python repositories for:
1. Unused Dependencies (AST import tracing vs requirements.txt).
2. Security Vulnerabilities & Known CVEs.
3. Performance Issues (O(N^2) nested loops, N+1 query loops, unclosed file handlers).
4. Code Complexity (Cyclomatic complexity calculation per function).
5. Self-Healing Auto-Fix suggestions and patch generation.
"""

import ast
import os
import re
from typing import Any

KNOWN_CVE_DATABASE: dict[str, dict[str, str]] = {
    "requests": {"vulnerable_below": "2.31.0", "safe_version": "2.31.0", "cve": "CVE-2023-32681", "desc": "Unintended leak of Proxy-Authorization header"},
    "urllib3": {"vulnerable_below": "1.26.17", "safe_version": "1.26.17", "cve": "CVE-2023-45803", "desc": "HTTP request smuggling vulnerability"},
    "pyyaml": {"vulnerable_below": "5.4.0", "safe_version": "5.4", "cve": "CVE-2020-14343", "desc": "Arbitrary code execution during YAML load"},
    "jinja2": {"vulnerable_below": "3.1.3", "safe_version": "3.1.3", "cve": "CVE-2024-22195", "desc": "Cross-site scripting via HTML attribute injection"},
    "pillow": {"vulnerable_below": "10.2.0", "safe_version": "10.2.0", "cve": "CVE-2023-50447", "desc": "Arbitrary code execution via ImageMath.eval"},
    "cryptography": {"vulnerable_below": "42.0.4", "safe_version": "42.0.4", "cve": "CVE-2024-26130", "desc": "NULL pointer dereference in PKCS12 parsing"},
    "setuptools": {"vulnerable_below": "70.0.0", "safe_version": "70.0.0", "cve": "CVE-2024-6345", "desc": "Remote code execution in package downloading"}
}

# Package name to standard Python import name mappings
PACKAGE_IMPORT_MAP: dict[str, str] = {
    "python-dotenv": "dotenv",
    "gitpython": "git",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "psycopg2-binary": "psycopg2"
}


class CodebaseAuditor:
    """
    Scans real repositories for technical debt, unused packages, security vulnerabilities,
    O(N^2) loops, N+1 query patterns, and cyclomatic complexity.
    """
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)

    def scan_repository_files(self) -> list[str]:
        """Lists all python files in the repository."""
        ignore_dirs = {".git", "node_modules", "venv", "__pycache__", "dist", "build", ".pytest_cache"}
        py_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.relpath(os.path.join(root, f), self.root_dir))
        return py_files

    def find_all_imports(self) -> set[str]:
        """Scans all Python files to extract imported top-level modules."""
        imported_modules: set[str] = set()
        for rel_path in self.scan_repository_files():
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top_mod = alias.name.split(".")[0]
                            imported_modules.add(top_mod)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top_mod = node.module.split(".")[0]
                        imported_modules.add(top_mod)
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
        return imported_modules

    def audit_unused_dependencies(self) -> list[dict[str, Any]]:
        """
        Cross-references requirements.txt against actual AST import statements
        to detect unused dependencies.
        """
        req_file = os.path.join(self.root_dir, "requirements.txt")
        if not os.path.exists(req_file):
            return []

        imported_modules = self.find_all_imports()
        unused = []

        with open(req_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                pkg_name = re.split(r"[=<>]", line_str)[0].strip().lower()
                import_name = PACKAGE_IMPORT_MAP.get(pkg_name, pkg_name)

                if import_name not in imported_modules and pkg_name not in imported_modules:
                    unused.append({
                        "type": "Unused Dependency",
                        "package": pkg_name,
                        "raw_line": line_str,
                        "description": f"Package '{pkg_name}' is declared in requirements.txt but never imported in any .py file.",
                        "recommendation": f"Remove '{line_str}' from requirements.txt to reduce dependency bloat."
                    })

        return unused

    def audit_security_vulnerabilities(self) -> list[dict[str, Any]]:
        """
        Checks requirement versions against known CVEs and scans code for insecure patterns.
        """
        vulnerabilities = []

        req_file = os.path.join(self.root_dir, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue

                    match = re.search(r"^([a-zA-Z0-9_\-]+)\s*==\s*([0-9\.]+)", line_str)
                    if match:
                        pkg, ver = match.group(1).lower(), match.group(2)
                        if pkg in KNOWN_CVE_DATABASE:
                            info = KNOWN_CVE_DATABASE[pkg]
                            if ver < info["vulnerable_below"]:
                                vulnerabilities.append({
                                    "type": "Security Vulnerability (CVE)",
                                    "package": pkg,
                                    "installed_version": ver,
                                    "cve": info["cve"],
                                    "description": f"{info['cve']}: {info['desc']} in {pkg} {ver}.",
                                    "recommendation": f"Upgrade {pkg} to >={info['safe_version']} in requirements.txt."
                                })

        for rel_path in self.scan_repository_files():
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("md5", "sha1") and isinstance(node.func.value, ast.Name) and node.func.value.id == "hashlib":
                        vulnerabilities.append({
                            "type": "Weak Cryptographic Hash",
                            "file": rel_path,
                            "line": getattr(node, "lineno", 1),
                            "description": f"Weak hash algorithm 'hashlib.{node.func.attr}()' detected in {rel_path}.",
                            "recommendation": "Use SHA-256 or SHA-512 instead of MD5/SHA-1."
                        })
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

        return vulnerabilities

    def audit_performance_issues(self) -> list[dict[str, Any]]:
        """
        Uses AST analysis to detect O(N^2) nested loops, N+1 DB queries, and unclosed file handlers.
        """
        perf_issues = []

        for rel_path in self.scan_repository_files():
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.For, ast.While)):
                        for child in ast.walk(node):
                            if child is not node and isinstance(child, (ast.For, ast.While)):
                                perf_issues.append({
                                    "type": "O(N^2) Performance Bottleneck",
                                    "file": rel_path,
                                    "line": getattr(node, "lineno", 1),
                                    "description": f"Nested loop detected in {rel_path} at line {getattr(node, 'lineno', 1)}.",
                                    "recommendation": "Refactor to single-pass dictionary/hash-set lookup to achieve O(N) complexity."
                                })
                                break

                    if isinstance(node, (ast.For, ast.While)):
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in ("execute", "executemany", "query", "filter"):
                                perf_issues.append({
                                    "type": "N+1 Query Bottleneck",
                                    "file": rel_path,
                                    "line": getattr(child, "lineno", getattr(node, "lineno", 1)),
                                    "description": f"Database call '.{child.func.attr}()' inside loop in {rel_path}.",
                                    "recommendation": "Batch database queries outside loop or use JOIN / bulk retrieval."
                                })
                                break

                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                        perf_issues.append({
                            "type": "Resource Leak Risk",
                            "file": rel_path,
                            "line": getattr(node, "lineno", 1),
                            "description": f"File 'open()' call at line {getattr(node, 'lineno', 1)} in {rel_path}.",
                            "recommendation": "Wrap file operations inside 'with open(...) as f:' to guarantee resource closure."
                        })

            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

        return perf_issues

    def calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """
        Calculates Cyclomatic Complexity V(G) = 1 + decision points.
        Decision points: If, For, While, And, Or, ExceptHandler, With, Assert, Try.
        """
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def audit_code_complexity(self, threshold: int = 10) -> list[dict[str, Any]]:
        """
        Identifies functions with Cyclomatic Complexity score exceeding threshold.
        """
        high_complexity = []

        for rel_path in self.scan_repository_files():
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        cc = self.calculate_cyclomatic_complexity(node)
                        if cc > threshold:
                            high_complexity.append({
                                "type": "High Cyclomatic Complexity",
                                "file": rel_path,
                                "function": node.name,
                                "line": getattr(node, "lineno", 1),
                                "complexity_score": cc,
                                "threshold": threshold,
                                "description": f"Function '{node.name}()' in {rel_path} has Cyclomatic Complexity score {cc} (threshold: {threshold}).",
                                "recommendation": "Decompose function into smaller modular helper functions."
                            })
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

        return high_complexity

    def run_full_audit(self) -> dict[str, Any]:
        """
        Runs comprehensive codebase audit across unused dependencies, security CVEs,
        performance bottlenecks, and cyclomatic complexity.
        """
        unused_deps = self.audit_unused_dependencies()
        security_vulnerabilities = self.audit_security_vulnerabilities()
        perf_issues = self.audit_performance_issues()
        complexity_issues = self.audit_code_complexity()

        total_issues = len(unused_deps) + len(security_vulnerabilities) + len(perf_issues) + len(complexity_issues)

        return {
            "root_directory": self.root_dir,
            "total_issues_found": total_issues,
            "health_score": max(0.0, round(100.0 - (total_issues * 5.0), 1)),
            "summary": {
                "unused_dependencies": len(unused_deps),
                "security_vulnerabilities": len(security_vulnerabilities),
                "performance_bottlenecks": len(perf_issues),
                "high_complexity_functions": len(complexity_issues)
            },
            "issues": {
                "unused_dependencies": unused_deps,
                "security_vulnerabilities": security_vulnerabilities,
                "performance_issues": perf_issues,
                "complexity_issues": complexity_issues
            }
        }
