"""
Dashboard API utilizing FastAPI.
Exposes real-time agent memory, task logs, sandbox telemetry, and repository overview data.
"""
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.engine import Engine

app = FastAPI(title="Autonomous Software Agent Dashboard API", version="1.0.0")

# Enable CORS for frontend dashboard queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global or single active Engine instance representing current workspace context
engine = Engine(workspace_path=".")

class TaskRequest(BaseModel):
    task: str

@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Retrieves real-time resource telemetry and current state."""
    return {
        "status": "idle" if not engine.current_task else "working",
        "current_task": engine.current_task,
        "iteration_count": engine.iteration_count,
        "system": {
            "cpu_usage_pct": psutil.cpu_percent(),
            "memory_usage_pct": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024)
        }
    }

@app.post("/api/task")
def start_task(req: TaskRequest) -> dict[str, Any]:
    """Triggers autonomous run for the specified user request."""
    try:
        res = engine.execute_task(req.task)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
def get_logs() -> list[dict[str, Any]]:
    """Returns historical run execution details."""
    return engine.logs

@app.get("/api/repository")
def get_repository() -> dict[str, Any]:
    """Exposes mapped project structural maps and AST data."""
    return engine.repo_analyzer.build_project_context()

@app.get("/api/memory")
def get_memory() -> list[dict[str, Any]]:
    """Returns stored preferences, keys, and past success profiles."""
    return engine.memory.get_all_knowledge()
