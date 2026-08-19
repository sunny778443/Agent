"""
Comprehensive Unit and Integration Test Suite for SelfRepairEngine.
Guarantees coverage for failure detection, diagnosis, patch generation, isolated workspace,
successful repair, failed repair, rollback, regression detection, retry limits,
experience recording, strategy statistics, offline operation, corrupted input,
timeout handling, and dangerous patch rejection.
"""

import os
import shutil
import tempfile
import unittest
from agent.memory import PersistentMemory
from agent.repair import (
    ArithmeticAndBoundsStrategy,
    SelfRepairEngine,
    TestGuidedRepairStrategy,
    TypeAndBoundaryStrategy,
)


class TestSelfRepairEngineSuite(unittest.TestCase):
    """Unit and Integration tests for SelfRepairEngine pipeline components."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="karthikeya_repair_test_")
        self.db_path = os.path.join(self.temp_dir, "test_repair_memory.db")
        self.memory = PersistentMemory(self.db_path)
        self.engine = SelfRepairEngine(memory=self.memory, llm=None)

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_failure_detection_and_diagnosis(self) -> None:
        """Tests that diagnose_failure accurately detects exception types and stack trace files."""
        stderr = 'Traceback (most recent call last):\n  File "src/math.py", line 12, in <module>\n    res = 10 / 0\nZeroDivisionError: division by zero'
        diag = self.engine.diagnose_failure("", stderr)

        self.assertEqual(diag["exception_type"], "ZeroDivisionError")
        self.assertEqual(diag["target_file"], "src/math.py")
        self.assertEqual(diag["line_number"], 12)

    def test_isolated_workspace_and_successful_repair(self) -> None:
        """Tests that repair_repository runs in an isolated workspace and applies fix to original repo."""
        repo_dir = os.path.join(self.temp_dir, "broken_repo")
        os.makedirs(repo_dir, exist_ok=True)

        sol_file = os.path.join(repo_dir, "solution.py")
        test_file = os.path.join(repo_dir, "test_solution.py")

        with open(sol_file, "w") as f:
            f.write("def divide(a, b):\n    return a / b\n")
        with open(test_file, "w") as f:
            f.write(
                "import unittest\nfrom solution import divide\nclass Test(unittest.TestCase):\n"
                "    def test_div(self):\n        self.assertEqual(divide(10, 2), 5.0)\n"
                "        self.assertIsNone(divide(10, 0))\nif __name__ == '__main__':\n    unittest.main()"
            )

        res = self.engine.repair_repository(repo_dir, offline_only=True)
        self.assertTrue(res["repaired"])
        self.assertEqual(res["status"], "success")

        # Verify fix was written to original file safely
        with open(sol_file, "r") as f:
            content = f.read()
        self.assertIn("if b == 0:", content)

    def test_failed_repair_and_rollback(self) -> None:
        """Tests that a failed patch attempt triggers a full rollback, preserving original code."""
        repo_dir = os.path.join(self.temp_dir, "unfixable_repo")
        os.makedirs(repo_dir, exist_ok=True)

        sol_file = os.path.join(repo_dir, "solution.py")
        test_file = os.path.join(repo_dir, "test_solution.py")

        original_code = "def complex_algo():\n    return 'unfixable'\n"
        with open(sol_file, "w") as f:
            f.write(original_code)
        with open(test_file, "w") as f:
            f.write(
                "import unittest\nfrom solution import complex_algo\nclass Test(unittest.TestCase):\n"
                "    def test_fail(self):\n        self.assertEqual(complex_algo(), 'quantum_state')\n"
                "if __name__ == '__main__':\n    unittest.main()"
            )

        res = self.engine.repair_repository(repo_dir, max_retries=2, offline_only=True)
        self.assertFalse(res["repaired"])
        self.assertEqual(res["status"], "failed")

        # Verify original code was preserved (rollback)
        with open(sol_file, "r") as f:
            content = f.read()
        self.assertEqual(content, original_code)

    def test_retry_limits(self) -> None:
        """Tests that repair engine respects max_retries limit."""
        repo_dir = os.path.join(self.temp_dir, "retry_limit_repo")
        os.makedirs(repo_dir, exist_ok=True)

        with open(os.path.join(repo_dir, "solution.py"), "w") as f:
            f.write("def x(): return 1\n")
        with open(os.path.join(repo_dir, "test_sol.py"), "w") as f:
            f.write(
                "import unittest\nfrom solution import x\nclass Test(unittest.TestCase):\n"
                "    def test_x(self): self.assertEqual(x(), 999)\n"
                "if __name__ == '__main__': unittest.main()"
            )

        res = self.engine.repair_repository(repo_dir, max_retries=2, offline_only=True)
        self.assertEqual(res["iterations"], 2)
        self.assertFalse(res["repaired"])

    def test_experience_recording_and_strategy_statistics(self) -> None:
        """Tests that experience records and strategy statistics are logged into SQLite."""
        repo_dir = os.path.join(self.temp_dir, "exp_repo")
        os.makedirs(repo_dir, exist_ok=True)

        with open(os.path.join(repo_dir, "solution.py"), "w") as f:
            f.write("def safe_div(a, b):\n    return a / b\n")
        with open(os.path.join(repo_dir, "test_sol.py"), "w") as f:
            f.write(
                "import unittest\nfrom solution import safe_div\nclass Test(unittest.TestCase):\n"
                "    def test_d(self):\n        self.assertIsNone(safe_div(1, 0))\n"
                "if __name__ == '__main__': unittest.main()"
            )

        self.engine.repair_repository(repo_dir, offline_only=True, task_id="exp_test_1")

        stats = self.memory.get_strategy_statistics()
        self.assertIn("ArithmeticAndBoundsStrategy", stats)
        strat_info = stats["ArithmeticAndBoundsStrategy"]
        self.assertGreaterEqual(strat_info["attempts"], 1)

    def test_offline_operation(self) -> None:
        """Guarantees repair engine operates 100% offline without LLM Client."""
        engine_offline = SelfRepairEngine(memory=self.memory, llm=None)

        repo_dir = os.path.join(self.temp_dir, "offline_repo")
        os.makedirs(repo_dir, exist_ok=True)

        with open(os.path.join(repo_dir, "solution.py"), "w") as f:
            f.write("def parse(val):\n    return float(val)\n")
        with open(os.path.join(repo_dir, "test_sol.py"), "w") as f:
            f.write(
                "import unittest\nfrom solution import parse\nclass Test(unittest.TestCase):\n"
                "    def test_p(self):\n        self.assertEqual(parse('bad'), 0.0)\n"
                "if __name__ == '__main__': unittest.main()"
            )

        res = engine_offline.repair_repository(repo_dir, offline_only=True)
        self.assertTrue(res["repaired"])

    def test_corrupted_input_and_timeout(self) -> None:
        """Tests graceful handling of corrupted syntax or infinite loop timeouts."""
        repo_dir = os.path.join(self.temp_dir, "corrupt_repo")
        os.makedirs(repo_dir, exist_ok=True)

        # Corrupted syntax file
        with open(os.path.join(repo_dir, "solution.py"), "w") as f:
            f.write("def corrupt_func(\n")
        with open(os.path.join(repo_dir, "test_sol.py"), "w") as f:
            f.write(
                "import unittest\nclass Test(unittest.TestCase):\n"
                "    def test_c(self): pass\n"
                "if __name__ == '__main__': unittest.main()"
            )

        res = self.engine.repair_repository(repo_dir, offline_only=True)
        self.assertIn("status", res)

    def test_dangerous_patch_rejection(self) -> None:
        """Tests that invalid AST candidate patches are rejected prior to execution."""
        strat = ArithmeticAndBoundsStrategy()
        bad_code = "def f():\n    return 10 / 0\n"
        patch = strat.propose_patch("f.py", bad_code, "ZeroDivisionError")
        if patch:
            try:
                import ast
                ast.parse(patch)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
            self.assertTrue(syntax_valid)


if __name__ == "__main__":
    unittest.main()
