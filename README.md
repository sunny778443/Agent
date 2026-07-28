# Autonomous Engineering Agent - Production System

This repository hosts a production-grade, modular, self-repairing autonomous AI agent designed from scratch. The system is split into a robust, secure, AST-driven Python backend engine and an elegant React-based analytics/execution dashboard.

---

## 🏗️ Core Architecture & Modular Layout

- **`agent/`**: Core software agent pipeline and modules.
  - `cli.py`: Interactive and direct command line driver.
  - `engine.py`: Central orchestrator tying planning, scanning, generation, verification, and repair.
  - `planner.py`: Multi-step risk assessor and scheduler.
  - `llm.py`: Production LLM interface with full Anthropic and OpenAI support plus offline simulation.
  - `memory.py`: Persistent SQLite knowledge and successful fix profiles database.
  - `repository.py`: Dependency graph builder, AST structure crawler, and affected file tracer.
  - `sandbox.py`: Resource-constrained Docker execution context runner.
  - `github.py`: Integrated Git and repository automation layer.
  - `repair.py`: Self-repair loop driving recursive repair logic until all tests succeed.
  - `testing.py`: Automated test skeletons and frameworks runner.
  - `security.py`: Guardrails preventing command injection, secrets leakage, and unsafe code constructs.
  - `static_analysis.py`: Seamless linting with Ruff and MyPy.
  - `dashboard_api.py`: FastAPI server feeding real-time metrics, logs, and AST states to the dashboard.

- **`dashboard/`**: React + Vite + Tailwind CSS frontend showing trace logging, live system specs, sandbox statuses, and database knowledge cards.

- **`tests/`**: Full unit and integration test suite guaranteeing complete component validation.

- **`docker/`**: Production-ready container setups for both runner and sandboxed targets.

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.12+
- Node.js & npm (for dashboard)
- Docker

### Backend and Agent setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute tasks from command-line:
   ```bash
   python -m agent.cli --task "Your task description here"
   ```
3. Spin up the Dashboard API:
   ```bash
   uvicorn agent.dashboard_api:app --reload --port 8000
   ```

### Dashboard Setup
1. Install packages and start Vite development client:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```
2. Access the visual workspace dashboard locally via: `http://localhost:3000`

---

## 🛡️ Security Guardrails & Verification Specs
- Commands are validated against dangerous patterns before running.
- Injected prompts are automatically blocked with safety exceptions.
- Hardcoded secrets and keys are automatically masked from source codes.
- Sandbox constraints enforce strict timeouts and containerized isolation.
