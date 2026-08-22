"""
Unit and Integration Test Suite for CodebaseAuditor module.
Tests unused dependency scanning, security CVE detection, O(N^2) performance analysis,
N+1 DB query detection, unclosed file handler detection, and cyclomatic complexity calculations.
"""

import os
import shutil
import tempfile
import unittest
from agent.auditor import CodebaseAuditor


class TestCodebaseAuditorSuite(unittest.TestCase):
    """Unit and Integration tests for CodebaseAuditor."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="karthikeya_audit_test_")
        self.auditor = CodebaseAuditor(root_dir=self.temp_dir)

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_unused_dependencies(self) -> None:
        """Tests that unused requirements listed in requirements.txt are accurately flagged."""
        req_path = os.path.join(self.temp_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("requests>=2.31.0\nunused-library>=1.0.0\n")

        src_path = os.path.join(self.temp_dir, "app.py")
        with open(src_path, "w") as f:
            f.write("import requests\nprint(requests.__name__)\n")

        unused = self.auditor.audit_unused_dependencies()
        self.assertEqual(len(unused), 1)
        self.assertEqual(unused[0]["package"], "unused-library")

    def test_audit_security_vulnerabilities(self) -> None:
        """Tests that known vulnerable dependency versions and weak cryptographic hashes are flagged."""
        req_path = os.path.join(self.temp_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("requests==2.20.0\n")  # Known vulnerable version below 2.31.0

        src_path = os.path.join(self.temp_dir, "crypto_app.py")
        with open(src_path, "w") as f:
            f.write("import hashlib\nhash_val = hashlib.md5(b'data').hexdigest()\n")

        vulnerabilities = self.auditor.audit_security_vulnerabilities()
        self.assertEqual(len(vulnerabilities), 2)

        cve_issue = next(v for v in vulnerabilities if v["type"] == "Security Vulnerability (CVE)")
        self.assertEqual(cve_issue["package"], "requests")
        self.assertEqual(cve_issue["cve"], "CVE-2023-32681")

        hash_issue = next(v for v in vulnerabilities if v["type"] == "Weak Cryptographic Hash")
        self.assertEqual(hash_issue["file"], "crypto_app.py")

    def test_audit_performance_issues(self) -> None:
        """Tests detection of O(N^2) nested loops, N+1 DB queries, and unclosed file handles."""
        src_path = os.path.join(self.temp_dir, "perf.py")
        with open(src_path, "w") as f:
            f.write(
                "def process_data(items, db_conn):\n"
                "    f = open('data.txt')\n"  # Unclosed file handle
                "    for x in items:\n"
                "        db_conn.execute('SELECT 1')\n"  # N+1 DB query
                "        for y in items:\n"  # O(N^2) nested loop
                "            print(x, y)\n"
            )

        perf_issues = self.auditor.audit_performance_issues()
        self.assertGreaterEqual(len(perf_issues), 3)

        issue_types = [issue["type"] for issue in perf_issues]
        self.assertIn("Resource Leak Risk", issue_types)
        self.assertIn("N+1 Query Bottleneck", issue_types)
        self.assertIn("O(N^2) Performance Bottleneck", issue_types)

    def test_audit_code_complexity(self) -> None:
        """Tests cyclomatic complexity calculations and threshold flagging."""
        src_path = os.path.join(self.temp_dir, "complex.py")
        # Build a function with decision points > 10
        code_lines = ["def complex_decision_engine(x):"]
        for i in range(12):
            code_lines.append(f"    if x == {i}: print({i})")
        code_lines.append("    return x\n")

        with open(src_path, "w") as f:
            f.write("\n".join(code_lines))

        complexity_issues = self.auditor.audit_code_complexity(threshold=10)
        self.assertEqual(len(complexity_issues), 1)
        self.assertEqual(complexity_issues[0]["function"], "complex_decision_engine")
        self.assertGreater(complexity_issues[0]["complexity_score"], 10)

    def test_run_full_audit(self) -> None:
        """Tests end-to-end full audit aggregation and health score computation."""
        req_path = os.path.join(self.temp_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("requests==2.20.0\nunused-pkg==1.0\n")

        src_path = os.path.join(self.temp_dir, "main.py")
        with open(src_path, "w") as f:
            f.write("import requests\nf = open('log.txt')\n")

        report = self.auditor.run_full_audit()
        self.assertIn("health_score", report)
        self.assertIn("summary", report)
        self.assertGreater(report["total_issues_found"], 0)


if __name__ == "__main__":
    unittest.main()
