"""
Command Line Interface for the Autonomous Agent.
Provides interactive query interfaces, audit execution, task runner, and financial trading bot interface.
"""
import argparse
import json
import sys

from agent.engine import Engine
from agent.trading_bot import TradingBot


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Software Engineering Agent & Financial Trading Bot")
    parser.add_argument("--task", type=str, help="The engineering task description to run")
    parser.add_argument("--audit", action="store_true", help="Run codebase audit for technical debt, security, and performance")
    parser.add_argument("--trade", type=str, help="Run financial web search, thinking, and trade execution query (e.g. 'Analyze NVDA stock and news')")
    parser.add_argument("--ticker", type=str, help="Optional stock ticker symbol for trading analysis (e.g. AAPL, NVDA, TSLA)")
    parser.add_argument("--workspace", type=str, default=".", help="Working directory pathway")

    args = parser.parse_args()

    if args.trade:
        print("=== Financial Web Search, Thinking & Trading Bot ===")
        print(f"Query: {args.trade}")
        bot = TradingBot()
        cycle_res = bot.execute_trading_cycle(args.trade, ticker=args.ticker)
        print("\n" + cycle_res["search_and_thought"]["answer"])
        print("\n=== Trade Execution Result ===")
        print(json.dumps(cycle_res["trade_execution"], indent=2))
        print("\n=== Portfolio Summary ===")
        print(json.dumps(cycle_res["portfolio"], indent=2))
        return

    engine = Engine(workspace_path=args.workspace)

    if args.audit:
        print(f"Running Codebase Audit on workspace: {args.workspace}")
        audit_res = engine.audit_codebase()
        print(json.dumps(audit_res, indent=2))
        return

    if not args.task:
        print("Please provide a task description using --task, run financial query via --trade, or run --audit. Examples:")
        print("  python -m agent.cli --trade 'Should I buy NVDA stock based on latest news?'")
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
