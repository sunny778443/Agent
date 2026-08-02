"""
Dashboard API utilizing FastAPI.
Exposes real-time agent memory, task logs, sandbox telemetry, and repository overview data.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import psutil
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
# We specify memory.db explicitly to avoid conflicts during runs
engine = Engine(workspace_path=".", db_path="memory.db")

class TaskRequest(BaseModel):
    task: str

@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Retrieves real-time resource telemetry and current state."""
    cpu = 0.0
    mem_pct = 0.0
    mem_avail = 0.0
    try:
        cpu = psutil.cpu_percent()
        mem_pct = psutil.virtual_memory().percent
        mem_avail = psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        pass

    return {
        "status": "idle" if not engine.current_task else "working",
        "current_task": engine.current_task,
        "iteration_count": engine.iteration_count,
        "system": {
            "cpu_usage_pct": cpu,
            "memory_usage_pct": mem_pct,
            "memory_available_mb": mem_avail
        }
    }

@app.post("/api/task")
def start_task(req: TaskRequest) -> Dict[str, Any]:
    """Triggers autonomous run for the specified user request."""
    try:
        res = engine.execute_task(req.task)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
def get_logs() -> List[Dict[str, Any]]:
    """Returns historical run execution details."""
    return engine.logs

@app.get("/api/repository")
def get_repository() -> Dict[str, Any]:
    """Exposes mapped project structural maps and AST data."""
    try:
        return engine.repo_analyzer.build_project_context()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory")
def get_memory() -> List[Dict[str, Any]]:
    """Returns stored preferences, keys, and past success profiles."""
    try:
        return engine.memory.get_all_knowledge()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
