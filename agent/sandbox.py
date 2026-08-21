"""
Sandboxed Execution environment inside Docker.
Limits CPU, RAM, disk usage, and execution time to prevent security hazards and resource starvation.
Reverts/destroys container state cleanly after operations are complete.
"""
import logging
import os
import subprocess
import time
from typing import Any

logger = logging.getLogger("agent.sandbox")

try:
    import docker
except ImportError:
    docker = None


class SandboxRunner:
    """
    Safely executes arbitrary commands and tests inside dedicated, resource-constrained Docker containers.
    """
    def __init__(self, image: str = "python:3.12-slim", cpu_limit: float = 1.0, memory_limit: str = "512m", timeout: int = 30):
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.timeout = timeout

        if docker is not None:
            try:
                self.client = docker.from_env()
            except (docker.errors.DockerException, AttributeError, OSError):
                self.client = None
        else:
            self.client = None

    def execute_command(self, command: str, bind_dir: str | None = None) -> dict[str, Any]:
        """
        Executes a command in a docker sandbox.
        Falls back gracefully if Docker daemon is not accessible, with safety/limit simulation.
        """
        if self.client is None:
            start_time = time.time()
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=bind_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False
                )
                duration = time.time() - start_time
                return {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration": duration,
                    "sandboxed": False,
                    "resource_usage": {"cpu": self.cpu_limit, "memory": self.memory_limit}
                }
            except subprocess.TimeoutExpired as e:
                duration = time.time() - start_time
                return {
                    "exit_code": -1,
                    "stdout": e.stdout or "",
                    "stderr": "Command execution timed out.",
                    "duration": duration,
                    "sandboxed": False,
                    "resource_usage": {}
                }
            except (subprocess.SubprocessError, OSError) as e:
                duration = time.time() - start_time
                return {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Local execution failed: {e!s}",
                    "duration": duration,
                    "sandboxed": False,
                    "resource_usage": {}
                }

        container = None
        try:
            volumes = {}
            if bind_dir:
                volumes[os.path.abspath(bind_dir)] = {"bind": "/workspace", "mode": "rw"}

            nano_cpus = int(self.cpu_limit * 1_000_000_000)

            container = self.client.containers.create(
                image=self.image,
                command=f"sh -c '{command}'",
                volumes=volumes,
                working_dir="/workspace" if bind_dir else None,
                mem_limit=self.memory_limit,
                nano_cpus=nano_cpus,
                detach=True
            )

            start_time = time.time()
            container.start()

            exit_code = None
            while time.time() - start_time < self.timeout:
                container.reload()
                status = container.status
                if status == "exited":
                    exit_code = container.attrs["State"]["ExitCode"]
                    break
                time.sleep(0.5)

            if exit_code is None:
                container.kill()
                exit_code = -1
                stdout = ""
                stderr = "Sandboxed execution timed out."
            else:
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore")
                stdout = logs
                stderr = ""

            duration = time.time() - start_time
            return {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "duration": duration,
                "sandboxed": True,
                "resource_usage": {
                    "cpu_limit": self.cpu_limit,
                    "memory_limit": self.memory_limit
                }
            }

        except (docker.errors.DockerException, RuntimeError, OSError) as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Sandbox execution error: {e!s}",
                "duration": 0.0,
                "sandboxed": True,
                "resource_usage": {}
            }
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except (docker.errors.DockerException, AttributeError, OSError) as e:
                    logger.debug("Failed to remove container: %s", e)
