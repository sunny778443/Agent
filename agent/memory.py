"""
Persistent and Experience Memory module using SQLite.
Stores historical actions, previous bugs, successful fixes, styles, preferences,
and multi-dimensional Experience Memory with semantic embedding similarity matching.
"""
import json
import math
import re
import sqlite3
from typing import Any


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two float vectors from scratch."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(y * y for y in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class PersistentMemory:
    """
    SQLite backed persistent database for storing agent knowledge, bug fixes, styles,
    and structured task experience memories with from-scratch semantic index matching.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT,
                    plan TEXT,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bug_fixes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bug_signature TEXT,
                    error_logs TEXT,
                    successful_patch TEXT,
                    project_context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT,
                    category TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experience_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    objective TEXT,
                    reasoning_steps TEXT,
                    tools_used TEXT,
                    code_changes TEXT,
                    success INTEGER,
                    execution_time REAL,
                    confidence REAL,
                    user_feedback TEXT,
                    lessons_learned TEXT,
                    reward REAL,
                    embedding TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def log_execution(self, task: str, plan: dict[str, Any], status: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO execution_history (task, plan, status) VALUES (?, ?, ?)",
                (task, json.dumps(plan), status)
            )
            conn.commit()
            return cursor.lastrowid or 0

    def store_bug_fix(self, bug_signature: str, error_logs: str, successful_patch: str, project_context: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bug_fixes (bug_signature, error_logs, successful_patch, project_context) VALUES (?, ?, ?, ?)",
                (bug_signature, error_logs, successful_patch, project_context)
            )
            conn.commit()

    def find_similar_fixes(self, error_text: str) -> list[dict[str, Any]]:
        """
        Retrieves matching successful fixes by scanning stored bug signatures and logs.
        """
        matches = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bug_fixes")
            rows = cursor.fetchall()
            for row in rows:
                sig = row["bug_signature"].lower()
                err = error_text.lower()

                sig_words = set(re.findall(r"\w+", sig))
                err_words = set(re.findall(r"\w+", err))
                common = sig_words.intersection(err_words)

                if sig in err or err in sig or len(common) >= 2:
                    matches.append({
                        "id": row["id"],
                        "bug_signature": row["bug_signature"],
                        "error_logs": row["error_logs"],
                        "successful_patch": row["successful_patch"],
                        "project_context": row["project_context"]
                    })
        return matches

    def store_preference(self, key: str, value: str, category: str = "preference") -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO project_knowledge (key, value, category) VALUES (?, ?, ?)",
                (key, value, category)
            )
            conn.commit()

    def get_preference(self, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM project_knowledge WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_knowledge(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM project_knowledge")
            return [dict(r) for r in cursor.fetchall()]

    def store_experience(
        self,
        objective: str,
        reasoning_steps: str,
        tools_used: str,
        code_changes: str,
        success: bool,
        execution_time: float,
        confidence: float | None,
        user_feedback: str,
        lessons_learned: str,
        reward: float,
        embedding: list[float]
    ) -> None:
        """Saves a multi-dimensional task experience with its computed semantic vector embed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experience_memory (
                    objective, reasoning_steps, tools_used, code_changes,
                    success, execution_time, confidence, user_feedback,
                    lessons_learned, reward, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    objective,
                    reasoning_steps,
                    tools_used,
                    code_changes,
                    1 if success else 0,
                    execution_time,
                    confidence,
                    user_feedback,
                    lessons_learned,
                    reward,
                    json.dumps(embedding)
                )
            )
            conn.commit()

    def get_strategy_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Calculates true statistical metrics for each execution strategy
        based purely on actual historical observations in SQLite experience logs.
        """
        stats = {}
        known_strategies = ["FastLinterAutoFix", "NeuralPromptContextualRepair"]
        for strat in known_strategies:
            stats[strat] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": None,
                "mean_reward": None,
                "variance": None,
                "status": "insufficient_data"
            }

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT tools_used, success, reward FROM experience_memory")
                rows = cursor.fetchall()

                grouped: dict[str, list[dict[str, Any]]] = {}
                for r in rows:
                    strat = r["tools_used"]
                    if not strat:
                        continue
                    if strat not in grouped:
                        grouped[strat] = []
                    grouped[strat].append({
                        "success": bool(r["success"]),
                        "reward": float(r["reward"])
                    })

                for strat, items in grouped.items():
                    attempts = len(items)
                    successes = sum(1 for item in items if item["success"])
                    failures = attempts - successes
                    success_rate = successes / attempts if attempts > 0 else 0.0
                    rewards = [item["reward"] for item in items]
                    mean_reward = sum(rewards) / attempts if attempts > 0 else 0.0

                    variance = 0.0
                    if attempts > 1:
                        variance = sum((r - mean_reward) ** 2 for r in rewards) / (attempts - 1)

                    stats[strat] = {
                        "attempts": attempts,
                        "successes": successes,
                        "failures": failures,
                        "success_rate": round(success_rate, 4),
                        "mean_reward": round(mean_reward, 4),
                        "variance": round(variance, 4),
                        "status": "calibrated" if attempts >= 3 else "insufficient_data"
                    }
        except sqlite3.Error:
            pass

        return stats

    def get_strategy_rankings(self) -> dict[str, float]:
        """
        Returns a simplified map of strategy names to their actual success rates.
        Returns empty dictionary if there is insufficient calibrated data.
        """
        stats = self.get_strategy_statistics()
        rankings = {}
        for strat, s in stats.items():
            if s["status"] == "calibrated" and s["success_rate"] is not None:
                rankings[strat] = s["success_rate"]
        return rankings

    def search_experiences_semantically(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """
        Searches previous experience records using from-scratch cosine similarity calculations
        of stored embedding vector projections.
        """
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experience_memory")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    stored_emb = json.loads(r["embedding"])
                    similarity = cosine_similarity(query_embedding, stored_emb)
                    results.append({
                        "id": r["id"],
                        "objective": r["objective"],
                        "reasoning_steps": r["reasoning_steps"],
                        "tools_used": r["tools_used"],
                        "code_changes": r["code_changes"],
                        "success": bool(r["success"]),
                        "execution_time": r["execution_time"],
                        "confidence": r["confidence"],
                        "user_feedback": r["user_feedback"],
                        "lessons_learned": r["lessons_learned"],
                        "reward": r["reward"],
                        "similarity": similarity,
                        "timestamp": r["timestamp"]
                    })
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
