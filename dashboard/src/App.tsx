import React, { useState, useEffect } from 'react';
import {
  Terminal, Shield, Play, RotateCcw, Cpu, HardDrive, RefreshCw,
  Layers, Database, Activity, Code, Compass, HelpCircle,
  ChevronRight, AlertTriangle, CheckCircle2, Sliders, Brain, MessageSquare
} from 'lucide-react';

interface SystemStats {
  cpu_usage_pct: number;
  memory_usage_pct: number;
  memory_available_mb: number;
}

interface StatusResponse {
  status: string;
  current_task: string | null;
  iteration_count: number;
  system: SystemStats;
}

interface LogEntry {
  timestamp: string;
  event_type: string;
  details: any;
}

interface RepoOverview {
  total_files: number;
  python_files: number;
  js_ts_files: number;
  docker_files: number;
  architecture_type: string;
}

interface MemoryKnowledge {
  id: number;
  key: string;
  value: string;
  category: string;
}

export default function App() {
  const [taskInput, setTaskInput] = useState('');
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [repo, setRepo] = useState<RepoOverview | null>(null);
  const [memory, setMemory] = useState<MemoryKnowledge[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Custom states for instructions
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [riskAssessment, setRiskAssessment] = useState<string>('N/A');
  const [currentGoalStep, setCurrentGoalStep] = useState<number>(0);

  const presets = [
    {
      title: "🐛 Debug & Self-Repair Division Bug",
      desc: "Scans repository for ZeroDivisionErrors, targets affected files, spins up a secure docker sandbox to replicate failures, and triggers the autonomous repair loop.",
      prompt: "Find and automatically repair division-by-zero vulnerability in mathematical codebases with pytest edge-case validations."
    },
    {
      title: "🛡️ Run Static Analysis & Lints",
      desc: "Invokes sandboxed Ruff and MyPy compilation checks over source workspace targets to enforce type compliance and clean coding standards.",
      prompt: "Execute static analysis, ruff lint checks, and run mypy compilation on local python modules to auto-repair formatting."
    },
    {
      title: "🧠 Run Cognitive Deep Learning Risk Scan",
      desc: "Feeds task sequences through our multi-head attention blocks and dependency networks to forecast engineering impact risk score dynamically.",
      prompt: "Evaluate the dynamic safety risk profile of implementing multi-layered database schema changes in backend core."
    },
    {
      title: "📦 Scaffold Production Code & Tests",
      desc: "Scaffolds a complete unit-tested file structure under target paths with complete robust code logic and matching automated test skeleton.",
      prompt: "Create a complete, robust user authentication helper class in src/auth_helper.py with pytest test-cases covering edge cases."
    }
  ];

  const fetchDashboardData = async () => {
    try {
      const statusRes = await fetch('/api/status');
      if (statusRes.ok) setStatus(await statusRes.json());

      const logsRes = await fetch('/api/logs');
      if (logsRes.ok) {
        const data = await logsRes.json();
        setLogs(data);
        // Dynamically extract cognitive/plan details if they exist in logs
        const planGen = data.find((l: any) => l.event_type === 'PLAN_GENERATED');
        if (planGen && planGen.details) {
          setRiskAssessment(`${Math.round(planGen.details.overall_risk_score * 100)}% - ${planGen.details.risk_assessment}`);
        }
      }

      const repoRes = await fetch('/api/repository');
      if (repoRes.ok) setRepo(await repoRes.json());

      const memoryRes = await fetch('/api/memory');
      if (memoryRes.ok) setMemory(await memoryRes.json());

      setApiError(null);
    } catch (err: any) {
      setApiError('Unable to connect to the agent backend API. Ensure uvicorn is running.');
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 4000);
    return () => clearInterval(interval);
  }, []);

  const triggerTask = async (promptText?: string) => {
    const finalPrompt = promptText || taskInput;
    if (!finalPrompt.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: finalPrompt }),
      });
      await fetchDashboardData();
    } catch (err) {
      setApiError('Failed to dispatch autonomous execution instructions.');
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (prompt: string) => {
    setTaskInput(prompt);
    setSelectedPreset(prompt);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-slate-950">
      {/* Top Banner */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-emerald-400 animate-pulse" />
          <div>
            <h1 className="text-lg font-bold tracking-tight flex items-center space-x-2">
              <span>Autonomous Agent Control Center</span>
              <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">Cognitive v2.5</span>
            </h1>
            <p className="text-xs text-slate-400">Instruct and dispatch autonomous software agents securely inside sandboxes</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          {status && (
            <div className="flex items-center space-x-2">
              <span className={`h-2 w-2 rounded-full ${status.status === 'working' ? 'bg-amber-400 animate-ping' : 'bg-emerald-400'}`}></span>
              <span className="text-xs font-mono font-bold text-slate-300">
                STATUS: {status.status.toUpperCase()}
              </span>
            </div>
          )}
          <button
            onClick={fetchDashboardData}
            className="p-2 text-slate-400 hover:text-white transition rounded-lg bg-slate-800 border border-slate-700 hover:border-slate-600"
            title="Refresh logs & telemetry"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-4 gap-6 overflow-auto max-w-[1600px] mx-auto w-full">
        {/* Error Notification */}
        {apiError && (
          <div className="xl:col-span-4 bg-red-950/30 border border-red-800/60 text-red-200 px-4 py-3 rounded-xl flex items-center justify-between animate-fadeIn">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-400" />
              <span className="text-sm font-medium">{apiError}</span>
            </div>
            <button onClick={() => setApiError(null)} className="text-xs font-semibold underline hover:text-white">Dismiss</button>
          </div>
        )}

        {/* ================= COLUMN 1 & 2: CONSOLE INSTRUCTION PANEL ================= */}
        <div className="xl:col-span-2 space-y-6 flex flex-col">
          {/* Main Agent Command Console */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden flex flex-col">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-indigo-500" />

            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-bold flex items-center space-x-2">
                <MessageSquare className="h-5 w-5 text-emerald-400" />
                <span>Command & Instruction Portal</span>
              </h2>
              <span className="text-xs text-slate-400 font-mono">Cognitive Assessment: {riskAssessment}</span>
            </div>

            <p className="text-xs text-slate-400 mb-4">
              Type free-form requests or pick a preset below. The agent will formulate planning steps, evaluate code dependency graphs via our internal neural attention layers, execute compilation checks inside resource-capped Docker sandboxes, and loop repairs if exceptions are raised.
            </p>

            <div className="space-y-4 flex-1">
              <div className="relative">
                <textarea
                  className="w-full h-36 bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all font-mono"
                  placeholder="Dispatch system instructions... E.g. 'Scaffold a safe matrix class with custom determinants, verify it with unit tests, and resolve failures.'"
                  value={taskInput}
                  onChange={(e) => setTaskInput(e.target.value)}
                />
              </div>

              <div className="flex items-center space-x-4">
                <button
                  onClick={() => triggerTask()}
                  disabled={loading || !taskInput.trim()}
                  className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-500 font-bold text-slate-950 py-3 px-6 rounded-xl transition-all flex items-center justify-center space-x-2 shadow-lg shadow-emerald-500/10 active:scale-95 cursor-pointer"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="h-5 w-5 animate-spin text-slate-950" />
                      <span>Agent Processing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 text-slate-950 fill-slate-950" />
                      <span>Dispatch Agent Instructions</span>
                    </>
                  )}
                </button>

                <button
                  onClick={() => { setTaskInput(''); setSelectedPreset(''); }}
                  className="p-3 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800 hover:border-slate-700 transition rounded-xl"
                  title="Clear parameters"
                >
                  <RotateCcw className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Preset Task Templates */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex-1">
            <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center space-x-2">
              <Compass className="h-4 w-4 text-sky-400" />
              <span>Instruction Presets & Blueprints</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {presets.map((preset, index) => (
                <div
                  key={index}
                  onClick={() => applyPreset(preset.prompt)}
                  className={`border p-4 rounded-xl cursor-pointer transition-all hover:bg-slate-800/40 hover:border-slate-600 flex flex-col ${
                    selectedPreset === preset.prompt ? 'border-emerald-500 bg-emerald-500/5' : 'border-slate-800 bg-slate-950/40'
                  }`}
                >
                  <span className="text-xs font-bold text-slate-200 mb-1 flex items-center space-x-1">
                    <span>{preset.title}</span>
                  </span>
                  <p className="text-[11px] text-slate-400 leading-relaxed mb-3 flex-1">{preset.desc}</p>
                  <div className="flex items-center text-[10px] text-emerald-400 font-semibold uppercase tracking-wider">
                    <span>Load Preset</span>
                    <ChevronRight className="h-3 w-3 ml-0.5" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ================= COLUMN 3: RUNTIME TRACE TRACING ================= */}
        <div className="xl:col-span-1 flex flex-col min-h-[450px]">
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl flex-1 flex flex-col overflow-hidden">
            <div className="border-b border-slate-800 px-5 py-4 bg-slate-900/40 flex items-center justify-between">
              <h2 className="text-sm font-bold flex items-center space-x-2">
                <Terminal className="h-4 w-4 text-amber-400" />
                <span>Execution Trace & Sandbox Telemetry</span>
              </h2>
              {status && status.status === 'working' && (
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                </span>
              )}
            </div>

            <div className="flex-1 p-5 overflow-y-auto space-y-4 bg-slate-950/40 font-mono text-xs max-h-[550px]">
              {logs.length > 0 ? (
                logs.map((log, index) => (
                  <div key={index} className="border-l border-slate-800 pl-3 py-1 hover:bg-slate-900/20 rounded transition-all">
                    <div className="flex items-center space-x-2 text-[10px] text-slate-500 mb-0.5">
                      <span>[{log.timestamp}]</span>
                      <span className="text-emerald-400 font-semibold">{log.event_type}</span>
                    </div>
                    <pre className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">
                      {JSON.stringify(log.details, null, 2)}
                    </pre>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-500 py-16 flex flex-col items-center justify-center space-y-2">
                  <Sliders className="h-8 w-8 text-slate-600 animate-spin" style={{ animationDuration: '6s' }} />
                  <p className="text-xs">Console is idle. Dispatch instructions to observe sandbox executions.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ================= COLUMN 4: METRICS & SYSTEM GRAPH ================= */}
        <div className="xl:col-span-1 space-y-6 flex flex-col">
          {/* Neural & Project Context Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
            <h2 className="text-sm font-bold mb-4 flex items-center space-x-2">
              <Layers className="h-4 w-4 text-sky-400" />
              <span>Cognitive Workspace Graph</span>
            </h2>
            {repo ? (
              <div className="space-y-3.5 text-xs">
                <div className="grid grid-cols-2 gap-3 font-mono">
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/60 text-center">
                    <span className="block text-[10px] text-slate-500 uppercase">Scanned files</span>
                    <span className="text-lg font-bold text-sky-400">{repo.total_files}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/60 text-center">
                    <span className="block text-[10px] text-slate-500 uppercase">Python modules</span>
                    <span className="text-lg font-bold text-emerald-400">{repo.python_files}</span>
                  </div>
                </div>

                <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/60">
                  <span className="block text-[10px] text-slate-500 uppercase font-mono mb-1">Architecture profile</span>
                  <span className="text-xs font-semibold text-slate-200">{repo.architecture_type}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">Project structure has not been indexed.</p>
            )}
          </div>

          {/* SQLite Memory Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col">
            <h2 className="text-sm font-bold mb-4 flex items-center space-x-2">
              <Database className="h-4 w-4 text-rose-400" />
              <span>Persistent SQLite Memory</span>
            </h2>
            <div className="space-y-3 overflow-y-auto max-h-[300px] flex-1">
              {memory.length > 0 ? (
                memory.map((item) => (
                  <div key={item.id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl text-[11px] font-mono leading-relaxed">
                    <div className="flex justify-between text-rose-400 font-semibold mb-1">
                      <span>KEY: {item.key}</span>
                      <span className="text-[9px] text-slate-500 uppercase">{item.category}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed">{item.value}</p>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-600 py-12 text-xs">
                  SQLite memory contains no registered heuristics.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
