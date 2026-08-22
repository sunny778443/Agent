"""
Command Line Interface for the Autonomous Agent.
Provides interactive query interfaces, audit execution, and task runner.
"""
import argparse
import json
import sys

from agent.engine import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Software Engineering Agent")
    parser.add_argument("--task", type=str, help="The engineering task description to run")
    parser.add_argument("--audit", action="store_true", help="Run codebase audit for technical debt, security, and performance")
    parser.add_argument("--workspace", type=str, default=".", help="Working directory pathway")

    args = parser.parse_args()

    engine = Engine(workspace_path=args.workspace)

    if args.audit:
        print(f"Running Codebase Audit on workspace: {args.workspace}")
        audit_res = engine.audit_codebase()
        print(json.dumps(audit_res, indent=2))
        return

    if not args.task:
        print("Please provide a task description using --task, or run --audit. Examples:")
        print("  python -m agent.cli --task 'Create a file in src/math.py that divides two integers'")
        print("  python -m agent.cli --audit")
        sys.exit(1)

    print(f"Starting autonomous agent workspace: {args.workspace}")
    print(f"Task: {args.task}")

    result = engine.execute_task(args.task)

    print("\nExecution Completed!")
    print(f"Status: {result.get('status')}")
    print(f"Iterations: {result.get('iterations')}")
    if result.get("reason"):
        print(f"Reason / Errors: {result.get('reason')}")


if __name__ == "__main__":
    main()
