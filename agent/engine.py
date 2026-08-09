"""
Core Engine Orchestrator module.
Ties together planning, repository parsing, security safety checks, test suites,
sandboxed runtimes, and the autonomous self-repair cycle with strategy selection, reflections, and rewards.
"""
import json
import os
import time
from typing import Any

from agent.cognitive import CodeCognitiveNetwork
from agent.github import GitHubManager
from agent.llm import LLMClient
from agent.memory import PersistentMemory
from agent.planner import Planner
from agent.repair import SelfRepairLoop
from agent.repository import RepositoryAnalyzer
from agent.security import SecurityManager
from agent.static_analysis import StaticAnalyzer
from agent.testing import AutomatedTester
from agent.user_understanding import UserUnderstandingModel


class Engine:
    """
    The orchestrator that runs the entire autonomous engineering workflow with cognitive decision metrics.
    """
    def __init__(self, workspace_path: str = ".", db_path: str = "memory.db"):
        self.workspace_path = os.path.abspath(workspace_path)
        self.llm = LLMClient()
        self.planner = Planner(self.llm)
        self.repo_analyzer = RepositoryAnalyzer(self.workspace_path)
        self.security = SecurityManager()
        self.static_analyzer = StaticAnalyzer(self.workspace_path)
        self.tester = AutomatedTester(self.workspace_path)
        self.memory = PersistentMemory(db_path)
        self.repair_loop = SelfRepairLoop(self.llm, self.memory)
        self.github = GitHubManager(self.workspace_path)
        self.cognitive_net = CodeCognitiveNetwork()
        self.user_model = UserUnderstandingModel()

        # State log for current run / session to expose to dashboard
        self.current_task: str | None = None
        self.current_plan: dict[str, Any] | None = None
        self.iteration_count = 0
        self.logs: list[dict[str, Any]] = []

    def log_event(self, event_type: str, details: Any) -> None:
        """Appends structured events for status tracking and dashboard visualization."""
        log_entry = {
            "timestamp": "now",
            "event_type": event_type,
            "details": details
        }
        self.logs.append(log_entry)

    def determine_target_file(self, task_description: str) -> str:
        """
        Uses repository context and LLM to dynamically select the most appropriate
        file target to modify/create instead of hardcoding 'src/main.py'.
        """
        files = self.repo_analyzer.scan_files()
        prompt = f"""
We have a local repository with the following files:
{json.dumps(files, indent=2)}

And the user wants to accomplish the following task:
"{task_description}"

Which single file path in the workspace should we create or modify to implement this?
Respond ONLY with the relative file path. No surrounding text, no formatting.
Example: src/math.py
"""
        target = self.llm.generate(prompt).strip()
        # Fallback to standard python file
        if not target or "/" not in target and not target.endswith((".py", ".ts", ".js")):
            return "src/main.py"
        return target

    def execute_task(self, task_description: str) -> dict[str, Any]:
        """
        Runs the full autonomous architecture pipeline:
        1. Pre-reasoning SQLite Memory Retrieval.
        2. User Emotion & Skill Profiling.
        3. Prompt Safety Scan.
        4. Plan generation & Risk/Confidence Assessment.
        5. Tool Selection, Strategy Ranking, & Code generation.
        6. Sandboxed compilation and testing.
        7. Post-task computational rewards and reflections.
        """
        start_time = time.perf_counter()
        self.current_task = task_description
        self.iteration_count = 0
        self.logs.clear()

        self.log_event("START_TASK", {"task": task_description})

        # 1. Pre-reasoning SQLite Memory Retrieval
        self.log_event("TOOL_SELECTION", {"tool": "PersistentMemory", "phase": "retrieval"})
        similar_fixes = self.memory.find_similar_fixes(task_description)
        memory_context = ""
        if similar_fixes:
            memory_context = f"Found {len(similar_fixes)} matching historical bug-fix profiles."
            for idx, fix in enumerate(similar_fixes[:2]):
                memory_context += f" Fix #{idx+1}: Signature='{fix['bug_signature']}', Patch='{fix['successful_patch'][:100]}'."
        else:
            memory_context = "No direct matching historical bug-fix profiles retrieved from database."

        self.log_event("MEMORY_RETRIEVED", {"context_summary": memory_context})

        # 2. User Emotion & Skill Profiling
        user_profile = self.user_model.profile_user_request(task_description)
        self.log_event("USER_PROFILE_DETECTED", user_profile)

        # 3. Input Safety Validation
        is_safe_prompt, prompt_msg = self.security.validate_prompt(task_description)
        if not is_safe_prompt:
            self.log_event("SECURITY_BLOCKED", {"message": prompt_msg})
            return {"status": "blocked", "reason": prompt_msg}

        # --- STRATEGY LEARNING RANKING ---
        strategy_rankings = self.memory.get_strategy_rankings()
        strategy_stats = self.memory.get_strategy_statistics()
        self.log_event("STRATEGY_LEARNING_RANKINGS", {"rankings": strategy_rankings})

        # Decide the reference strategy for planning confidence calibration
        best_strategy = "NeuralPromptContextualRepair"
        if strategy_rankings:
            best_strategy = max(strategy_rankings, key=strategy_rankings.get)

        success_rate = strategy_rankings.get(best_strategy)
        sample_count = strategy_stats.get(best_strategy, {}).get("attempts", 0)

        # 4. Plan Generation & Risk/Confidence Assessment
        self.log_event("TOOL_SELECTION", {"tool": "Planner", "phase": "reasoning"})
        plan = self.planner.create_execution_plan_with_context(
            task_description,
            memory_context=memory_context,
            historical_success_rate=success_rate,
            sample_count=sample_count
        )
        self.current_plan = plan

        # Integrate Cognitive NN prediction to refine the risk assessment score dynamically
        cognitive_risk = self.cognitive_net.predict_task_risk(task_description)
        plan["overall_risk_score"] = float(cognitive_risk)
        plan["risk_assessment"] = "High Risk (Neural Class)" if cognitive_risk > 0.6 else "Standard Cognitive Risk"

        self.log_event("PLAN_GENERATED", plan)
        self.memory.log_execution(task_description, plan, "In Progress")

        # 5. Tool Selection & Code Generation
        self.log_event("TOOL_SELECTION", {"tool": "RepositoryAnalyzer", "phase": "targeting"})
        target_file = self.determine_target_file(task_description)
        self.log_event("TARGET_IDENTIFIED", {"target_file": target_file})

        target_path = os.path.join(self.workspace_path, target_file)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Generate code from LLM
        self.log_event("TOOL_SELECTION", {"tool": "LLMClient", "phase": "generation"})
        prompt = f"Generate complete, robust production-grade code to satisfy this task: {task_description}. Style guide parameters: {user_profile.get('style_guide', '')}"
        generated_code = self.llm.generate(prompt)

        # Sanitize and validate
        sanitized_code, secrets_removed = self.security.sanitize_code(generated_code)
        is_safe_code, code_msg = self.security.validate_code_safety(sanitized_code)

        if not is_safe_code:
            self.log_event("SECURITY_BLOCKED", {"message": code_msg})
            return {"status": "blocked", "reason": code_msg}

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(sanitized_code)
        self.log_event("WRITE_CODE", {"filepath": target_file, "secrets_removed": secrets_removed})

        # Evaluate the dynamic project graph structures with our neural node mapping
        try:
            self.log_event("TOOL_SELECTION", {"tool": "CodeCognitiveNetwork", "phase": "graph_analysis"})
            file_contents = {}
            for f in self.repo_analyzer.scan_files():
                if f.endswith(".py") and os.path.exists(os.path.join(self.workspace_path, f)):
                    with open(os.path.join(self.workspace_path, f), "r", encoding="utf-8") as file_read:
                        file_contents[f] = file_read.read()
            dep_graph = self.repo_analyzer.build_dependency_graph()
            cognitive_metrics = self.cognitive_net.process_project_dependency_graph(file_contents, dep_graph)
            self.log_event("NEURAL_CODEBASE_ANALYSIS", {"metrics": cognitive_metrics})
        except Exception as cog_err:
            self.log_event("NEURAL_ANALYSIS_ERROR", {"detail": str(cog_err)})

        # Generate test skeletons automatically for the file
        if target_file.endswith(".py"):
            self.log_event("TOOL_SELECTION", {"tool": "AutomatedTester", "phase": "test_generation"})
            generated_test_file = self.tester.generate_tests(target_file, sanitized_code)
            self.log_event("TEST_SKELETON_GENERATED", {"test_file": generated_test_file})

        # 6. Verification and Sandboxed Self-Repair Loop (up to 3 iterations)
        max_iterations = 3
        success = False
        error_logs = ""

        reward_score = 0.0
        used_strategy = "NeuralPromptContextualRepair"

        for iteration in range(1, max_iterations + 1):
            self.iteration_count = iteration
            self.log_event("VERIFICATION_ATTEMPT", {"iteration": iteration})

            # Run Lint / Static Analysis & tests securely inside Docker Sandbox
            self.log_event("TOOL_SELECTION", {"tool": "SandboxRunner", "phase": "validation"})
            lint_res = self.tester.sandbox.execute_command(f"ruff check {target_file}", bind_dir=self.workspace_path)
            type_res = self.tester.sandbox.execute_command(f"mypy {target_file} --ignore-missing-imports", bind_dir=self.workspace_path)
            test_res = self.tester.run_tests_sandboxed(framework="pytest")

            lint_success = (lint_res["exit_code"] == 0)
            type_success = (type_res["exit_code"] == 0)
            test_success = (test_res["exit_code"] == 0)

            if lint_success and type_success and test_success:
                success = True
                self.log_event("VERIFICATION_SUCCESS", {"iteration": iteration})
                reward_score += 15.0 # +15 tests passed
                break

            # Penalize failures
            reward_score -= 15.0 # -15 failed build/lints
            if not test_success:
                reward_score -= 10.0 # -10 failed tests/exceptions

            # Capture issues and choose best repair strategy
            error_logs = f"Sandbox Lint Success: {lint_success}. Sandbox Type Success: {type_success}. Sandbox Test Success: {test_success}.\n"
            if not test_success:
                error_logs += f"Sandbox Test Output:\n{test_res['stderr'] or test_res['stdout']}\n"
            if not lint_success:
                error_logs += f"Sandbox Lint Output:\n{lint_res['stderr'] or lint_res['stdout']}\n"
            if not type_success:
                error_logs += f"Sandbox Type Output:\n{type_res['stderr'] or type_res['stdout']}\n"

            self.log_event("VERIFICATION_FAILURE", {"iteration": iteration, "errors": error_logs})

            # --- DECIDE OPTIMAL STRATEGY before acting ---
            # Prefer FastLinterAutoFix if it's purely formatting and has higher historical success rate
            linter_success_rate = strategy_rankings.get("FastLinterAutoFix", 0.5)
            neural_success_rate = strategy_rankings.get("NeuralPromptContextualRepair", 0.5)

            if not lint_success and type_success and test_success and linter_success_rate > neural_success_rate:
                # STRATEGY A: Fast local ruff check --fix
                used_strategy = "FastLinterAutoFix"
                self.log_event("STRATEGY_SELECTED", {"strategy": "FastLinterAutoFix", "reason": "Ruff formatting issue detected and ranked higher"})
                self.static_analyzer.repair_lint_issues(target_file, linter="ruff")
            else:
                # STRATEGY B: Neural contextual prompt repair loop
                used_strategy = "NeuralPromptContextualRepair"
                self.log_event("STRATEGY_SELECTED", {"strategy": "NeuralPromptContextualRepair", "reason": "Structural logic or test failing"})
                repaired_code = self.repair_loop.run_repair_iteration(target_file, sanitized_code, error_logs)
                sanitized_code, _ = self.security.sanitize_code(repaired_code)

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(sanitized_code)

            self.log_event("APPLIED_REPAIR", {"filepath": target_file})

        duration = time.perf_counter() - start_time

        # Reward bonuses
        if success:
            reward_score += 10.0 # +10 bug fixed
        if duration < 1.5:
            reward_score += 5.0 # +5 faster execution

        # 6. Post-task Self-Reflection
        reflection = {
            "task": task_description,
            "target_file": target_file,
            "verification_success": success,
            "total_repair_iterations": self.iteration_count,
            "plan_confidence": plan.get("confidence_score"),
            "duration_seconds": duration,
            "reward_score": reward_score,
            "what_went_well": "Sandbox tests and lint compilations passed cleanly" if success else "Initial code generated, but sandbox checks failed",
            "what_failed": "None" if success else "Failed to pass automated sandboxed tests within iterations",
            "better_strategy_suggestion": "The current tool selection was optimal" if success else "Consider writing intermediate mock unit-tests first to isolate targets",
            "what_should_be_remembered": f"Task file {target_file} resolved cleanly with patch."
        }

        reflection_key = f"reflection_{int(time.time())}"
        self.memory.store_preference(reflection_key, json.dumps(reflection), category="reflections")
        self.log_event("POST_TASK_REFLECTION", reflection)

        # 7. Experience Memory Insertion with from-scratch embeddings
        seq = self.cognitive_net.encode_text_sequence(task_description)
        if seq:
            # Simple averaged token embedding from scratch
            avg_emb = [sum(col) / len(seq) for col in zip(*seq)]
            # Ensure it fits net.embed_dim (e.g. pad or truncate)
            while len(avg_emb) < self.cognitive_net.embed_dim:
                avg_emb.append(0.0)
            avg_emb = avg_emb[:self.cognitive_net.embed_dim]
        else:
            avg_emb = [0.0] * self.cognitive_net.embed_dim

        self.memory.store_experience(
            objective=task_description,
            reasoning_steps=json.dumps(plan.get("steps", [])),
            tools_used=used_strategy,
            code_changes=sanitized_code[:500],
            success=success,
            execution_time=duration,
            confidence=plan.get("confidence_score"),
            user_feedback=f"Analyzed emotion={user_profile.get('skill_level')}",
            lessons_learned=reflection["what_should_be_remembered"],
            reward=reward_score,
            embedding=avg_emb
        )

        if success:
            self.memory.store_bug_fix("successful-task-fix", "No errors", sanitized_code, target_file)

            # Safe branch and commit via GitHubManager
            try:
                # Checkout a dynamic task branch and commit changes securely
                branch_name = "agent-task-branch"
                self.github.create_branch(branch_name)
                commit_sha = self.github.commit_changes(f"feat: implement solutions for {task_description[:50]}")
                self.log_event("GIT_COMMIT_SUCCESS", {"branch": branch_name, "commit_sha": commit_sha})
            except Exception as git_err:
                self.log_event("GIT_COMMIT_SKIPPED", {"detail": str(git_err)})

            self.log_event("TASK_COMPLETE", {"status": "success"})
            return {"status": "success", "file": target_file, "iterations": self.iteration_count, "reward": reward_score}
        else:
            self.log_event("TASK_FAILED", {"status": "failed", "final_error": error_logs})
            return {"status": "failed", "reason": error_logs, "iterations": self.iteration_count, "reward": reward_score}
