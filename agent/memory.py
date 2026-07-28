"""
Persistent Memory module using SQLite.
Stores historical actions, previous bugs, successful fixes, styles, and configurations.
"""
import json
import re
import sqlite3
from typing import Any


class PersistentMemory:
    """
    SQLite backed persistent database for storing agent knowledge, bug fixes, styles, and preference matches.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table to store execution steps and actions taken
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT,
                    plan TEXT,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table to store past bugs and successful repairs/fixes
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
            # Table to store generalized project knowledge, styles, and preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT,
                    category TEXT
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
                logs = row["error_logs"].lower()
                err = error_text.lower()

                sig_words = set(re.findall(r'\w+', sig))
                err_words = set(re.findall(r'\w+', err))
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
