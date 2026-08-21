# Project Karthikeya - Modular AI Engineering Agent Framework

This repository hosts a modular, self-repairing autonomous AI software engineering agent framework designed from scratch. The system is split into an AST-driven Python backend engine, a persistent SQLite experience memory layer, secure Docker-sandboxed execution scripts, and an elegant React-based telemetry/execution dashboard.

---

## ⚠️ Realistic Technical Bounds & Limitations (Brutally Honest Pass)

Project Karthikeya is built strictly from first principles with zero external machine learning dependencies (no NumPy, PyTorch, or TensorFlow). For educational transparency and engineering precision, please note the following technical constraints:
1. **Linguistic Emotion-Estimation (NOT human-level AI)**: The `UserUnderstandingModel` is a heuristic regex and word-intersection keyword pattern matching system with negation context logic. It estimates potential user feelings and skill profiles, but does NOT possess genuine emotional intelligence or human-level cognitive understanding.
2. **Toy Generative Representation (NOT a photographic generator)**: The `GenerativeFaceNetwork` in `agent/cognitive.py` is a toy 64-dimensional Variational Autoencoder (VAE) representing and reconstructing 8x8 flattened pixel intensities. It is designed to demonstrate backpropagation and latent space reparameterization mathematical correctness, and CANNOT render high-resolution photographic images of human faces.
3. **Deterministic Seed Bounds**: Random state shuffles in the pipeline utilize epoch-based custom RNG inputs (`random.Random(seed + epoch)`) to guarantee determinism, which is independent of global system clocks.
4. **Computational Metrics**: Confidence calibration and strategies are assessed via the Brier Score and Expected Calibration Error (ECE) algorithms from scratch using Wilson Score confidence intervals.
5. **Static Pattern Security (NOT an absolute security boundary)**: The `SecurityManager` uses AST parsing and command tokenization filters to catch common unsafe patterns (e.g., `eval`, `exec`, `os.system`). It serves as a first-line defensive static filter, not an unbreakable security boundary. True isolation is provided by Docker sandbox runtimes.

---

## 🏗️ Core Architecture & Modular Layout

- **`agent/`**: Core software agent pipeline and modules.
  - `cli.py`: Interactive and direct command line driver.
  - `engine.py`: Central orchestrator tying planning, scanning, generation, verification, and repair.
  - `planner.py`: Multi-step risk assessor and scheduler.
  - `llm.py`: LLM interface that defaults to offline mock mode and supports OpenAI and Anthropic when `LLM_PROVIDER` and API key environment variables are set.
  - `memory.py`: Persistent SQLite knowledge and successful fix profiles database.
  - `repository.py`: Dependency graph builder, AST structure crawler, and affected file tracer.
  - `sandbox.py`: Resource-constrained Docker execution context runner.
  - `github.py`: Integrated Git and repository automation layer.
  - `repair.py`: Self-repair engine driving workspace-isolated recursive repair loops and rollbacks.
  - `testing.py`: Automated test skeletons and frameworks runner.
  - `security.py`: AST-based code inspection and command pattern filter.
  - `static_analysis.py`: Seamless linting with Ruff and MyPy.
  - `user_understanding.py`: Transparent emotion and skill profiling pipeline.
  - `calibration.py`: Brier Score and ECE calibration validation engine.
  - `dashboard_api.py`: FastAPI server feeding real-time metrics, logs, and AST states to the dashboard.

- **`dashboard/`**: React + Vite + Tailwind CSS frontend showing trace logging, live system specs, sandbox statuses, and database knowledge cards.

- **`tests/`**: Full unit, integration, self-repair, and benchmark test suite.

- **`docker/`**: Production-ready container setups for both runner and sandboxed targets.

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.12+
- Node.js & npm (for dashboard)
- Docker (optional, for isolated sandbox execution)

### LLM Provider Setup
By default, Karthikeya operates in **offline mock mode** without requiring external API keys.
To enable live AI generation with LLM providers, set the appropriate environment variables:
```bash
# For OpenAI
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

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
- AST-based inspection blocks dangerous function calls (`eval`, `exec`, `os.system`, `subprocess(shell=True)`).
- Commands are tokenized via `shlex` and filtered against forbidden executables and destructive flags.
- Injected prompt patterns are automatically detected and blocked.
- Hardcoded secrets and API keys are masked prior to file writes.
- Containerized Docker sandboxes enforce resource and network isolation.
