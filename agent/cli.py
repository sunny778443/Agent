"""
Command Line Interface for the Autonomous Agent.
Provides interactive or direct query interfaces.
"""
import argparse
import sys

from agent.engine import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Software Engineering Agent")
    parser.add_argument("--task", type=str, help="The engineering task description to run")
    parser.add_argument("--workspace", type=str, default=".", help="Working directory pathway")

    args = parser.parse_args()

    if not args.task:
        print("Please provide a task description using --task. Examples:")
        print("  python -m agent.cli --task 'Create a file in src/math.py that divides two integers with proper error handling'")
        sys.exit(1)

    print(f"Starting autonomous agent workspace: {args.workspace}")
    print(f"Task: {args.task}")

    engine = Engine(workspace_path=args.workspace)
    result = engine.execute_task(args.task)

    print("\nExecution Completed!")
    print(f"Status: {result.get('status')}")
    print(f"Iterations: {result.get('iterations')}")
    if result.get("reason"):
        print(f"Reason / Errors: {result.get('reason')}")

if __name__ == "__main__":
    main()
