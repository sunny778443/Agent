"""
Performance Benchmarking suite for Project Karthikeya Phase 2.
Measures latency profiles for AST crawlers, SQLite memory matching, and neural network inference passes.
"""

import os
import time
import unittest

from agent.cognitive import CodeCognitiveNetwork
from agent.memory import PersistentMemory
from agent.repository import RepositoryAnalyzer


class TestPerformanceBenchmarks(unittest.TestCase):
    """
    Measures latency profiles of critical agent components under realistic loads
    to ensure highly optimized and responsive autonomous performance.
    """
    def setUp(self) -> None:
        self.db_path = "benchmark_memory.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.memory = PersistentMemory(self.db_path)
        self.analyzer = RepositoryAnalyzer(".")
        self.net = CodeCognitiveNetwork(vocab_size=128, embed_dim=16)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_repository_crawling_and_ast_parsing_benchmark(self) -> None:
        """Benchmarks repository scan and file dependencies tracing latencies."""
        start_time = time.perf_counter()
        files = self.analyzer.scan_files()
        crawling_duration = time.perf_counter() - start_time

        self.assertIsInstance(files, list)
        print(f"\n[Benchmark] Repository Crawling Speed: parsed {len(files)} files in {crawling_duration * 1000:.3f} ms")

        # Benchmark dependency graph build
        start_time = time.perf_counter()
        graph = self.analyzer.build_dependency_graph()
        graph_duration = time.perf_counter() - start_time
        self.assertIsInstance(graph, dict)
        print(f"[Benchmark] AST Dependency Graph Build Speed: {graph_duration * 1000:.3f} ms")

    def test_sqlite_memory_retrieval_benchmark(self) -> None:
        """Benchmarks SQL queries and word-intersection matching speeds under pressure."""
        # Seed memory with mock bug fixes
        for i in range(100):
            self.memory.store_bug_fix(
                bug_signature=f"ExceptionType{i}: custom error signature {i}",
                error_logs=f"Traceback error log for testing matching of item {i}",
                successful_patch=f"def fix_{i}(): return {i}"
            )

        start_time = time.perf_counter()
        matches = self.memory.find_similar_fixes("custom error signature 42")
        retrieval_duration = time.perf_counter() - start_time

        self.assertTrue(len(matches) >= 1)
        print(f"[Benchmark] SQLite Word-Intersection Retrieval (100 records): {retrieval_duration * 1000:.3f} ms")

    def test_cognitive_network_inference_benchmark(self) -> None:
        """Benchmarks forward propagation and attention layer pool latencies."""
        task = "refactor production database schemas and resolve missing security authentication lints"

        start_time = time.perf_counter()
        risk = self.net.predict_task_risk(task)
        inference_duration = time.perf_counter() - start_time

        self.assertTrue(0.0 <= risk <= 1.0)
        print(f"[Benchmark] Cognitive Neural Inference Pass: {inference_duration * 1000:.3f} ms")

if __name__ == "__main__":
    unittest.main()
