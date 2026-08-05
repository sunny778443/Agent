import React, { useState, useEffect } from 'react';
import {
  Terminal, Shield, Play, RotateCcw, Cpu, HardDrive, RefreshCw,
  Layers, Database, Activity, Code, Compass, HelpCircle,
  ChevronRight, AlertTriangle, CheckCircle2, Sliders, Brain, MessageSquare,
  BookOpen, Star, TrendingDown, Target, Zap, Smile
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

interface SelfReflection {
  task: string;
  target_file: string;
  verification_success: boolean;
  total_repair_iterations: number;
  plan_confidence: number;
  timestamp: number;
  reward_score: number;
  what_went_well: string;
  better_strategy_suggestion: string;
}

interface ExperienceRecord {
  id: number;
  objective: string;
  tools_used: string;
  success: boolean;
  execution_time: number;
  confidence: number;
  reward: number;
  lessons_learned: string;
}

interface TrainingMetric {
  epoch: number;
  learning_rate: number;
  train_loss: number;
  train_mae: number;
  val_loss: number;
  val_mae: number;
}

export default function App() {
  const [taskInput, setTaskInput] = useState('');
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [repo, setRepo] = useState<RepoOverview | null>(null);
  const [memory, setMemory] = useState<MemoryKnowledge[]>([]);
  const [reflections, setReflections] = useState<SelfReflection[]>([]);
  const [experiences, setExperiences] = useState<ExperienceRecord[]>([]);
  const [strategyRankings, setStrategyRankings] = useState<Record<string, number>>({});
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Custom states for instructions
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [riskAssessment, setRiskAssessment] = useState<string>('N/A');
  const [confidenceScore, setConfidenceScore] = useState<number>(0.85);

  // User emotion estimates
  const [emotionProfile, setEmotionProfile] = useState<any>({
    frustration: 0.1,
    confusion: 0.1,
    urgency: 0.1,
    skill_level: "standard"
  });

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

        // Extract cognitive/plan details
        const planGen = data.find((l: any) => l.event_type === 'PLAN_GENERATED');
        if (planGen && planGen.details) {
          setRiskAssessment(`${Math.round(planGen.details.overall_risk_score * 100)}% - ${planGen.details.risk_assessment}`);
          if (planGen.details.confidence_score) {
            setConfidenceScore(planGen.details.confidence_score);
          }
        }

        // Extract user profile detected event
        const userProf = data.find((l: any) => l.event_type === 'USER_PROFILE_DETECTED');
        if (userProf && userProf.details) {
          setEmotionProfile(userProf.details);
        }
      }

      const repoRes = await fetch('/api/repository');
      if (repoRes.ok) setRepo(await repoRes.json());

      const memoryRes = await fetch('/api/memory');
      if (memoryRes.ok) setMemory(await memoryRes.json());

      const reflectionsRes = await fetch('/api/reflections');
      if (reflectionsRes.ok) setReflections(await reflectionsRes.json());

      const experiencesRes = await fetch('/api/experiences');
      if (experiencesRes.ok) setExperiences(await experiencesRes.json());

      const strategiesRes = await fetch('/api/strategy_rankings');
      if (strategiesRes.ok) setStrategyRankings(await strategiesRes.json());

      const metricsRes = await fetch('/api/training_metrics');
      if (metricsRes.ok) setTrainingMetrics(await metricsRes.json());

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
              <span>Project Karthikeya Adaptive Agent Portal</span>
              <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">Phase 3 Cognition</span>
            </h1>
            <p className="text-xs text-slate-400">Computational Reward Optimization & Multi-Paradigm Reinforce Engine</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          {status && (
            <div className="flex items-center space-x-2 bg-slate-950/60 border border-slate-800 px-3 py-1.5 rounded-lg">
              <span className={`h-2.5 w-2.5 rounded-full ${status.status === 'working' ? 'bg-amber-400 animate-ping' : 'bg-emerald-400 animate-pulse'}`}></span>
              <span className="text-xs font-mono font-bold text-slate-300">
                STATUS: {status.status.toUpperCase()}
              </span>
            </div>
          )}
          <button
            onClick={fetchDashboardData}
            className="p-2 text-slate-400 hover:text-white transition rounded-lg bg-slate-800 border border-slate-700 hover:border-slate-600 cursor-pointer"
            title="Refresh logs & telemetry"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-4 gap-6 overflow-auto max-w-[1700px] mx-auto w-full">
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

        {/* ================= COLUMN 1: CONSOLE INSTRUCTION PANEL ================= */}
        <div className="xl:col-span-1 space-y-6 flex flex-col">
          {/* Main Agent Command Console */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl relative overflow-hidden flex flex-col transition-all hover:border-slate-700">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-indigo-500" />

            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-bold flex items-center space-x-2">
                <MessageSquare className="h-4.5 w-4.5 text-emerald-400" />
                <span>Command Instruction Portal</span>
              </h2>
            </div>

            <textarea
              className="w-full h-28 bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all font-mono mb-4"
              placeholder="Dispatch instructions... E.g. 'Build a safe matrix division module with tests.'"
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
            />

            {/* Dynamic Emotion Profile Card */}
            <div className="bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl space-y-2 mb-4">
              <div className="flex items-center justify-between text-[10px] uppercase font-mono text-slate-400 border-b border-slate-800 pb-1.5 mb-1">
                <span className="flex items-center space-x-1">
                  <Smile className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Emotion Profile</span>
                </span>
                <span className="text-indigo-400 font-bold">Skill: {emotionProfile.skill_level}</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono text-center">
                <div className="bg-slate-900/60 p-1.5 rounded">
                  <span>FRUST:</span>
                  <span className="block font-bold text-slate-200">{Math.round(emotionProfile.frustration * 100)}%</span>
                </div>
                <div className="bg-slate-900/60 p-1.5 rounded">
                  <span>CONF:</span>
                  <span className="block font-bold text-slate-200">{Math.round(emotionProfile.confusion * 100)}%</span>
                </div>
                <div className="bg-slate-900/60 p-1.5 rounded">
                  <span>URG:</span>
                  <span className="block font-bold text-slate-200">{Math.round(emotionProfile.urgency * 100)}%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => triggerTask()}
                disabled={loading || !taskInput.trim()}
                className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-500 font-bold text-slate-950 py-2.5 px-4 rounded-xl transition-all flex items-center justify-center space-x-2 shadow-lg shadow-emerald-500/10 cursor-pointer text-xs"
              >
                {loading ? (
                  <RefreshCw className="h-4 w-4 animate-spin text-slate-950" />
                ) : (
                  <>
                    <Play className="h-4 w-4 text-slate-950 fill-slate-950" />
                    <span>Dispatch Agent</span>
                  </>
                )}
              </button>
              <button
                onClick={() => { setTaskInput(''); setSelectedPreset(''); }}
                className="p-2.5 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800 hover:border-slate-700 transition rounded-xl cursor-pointer"
                title="Clear parameters"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Strategy Rankings */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl transition-all hover:border-slate-700">
            <h3 className="text-xs font-bold text-indigo-400 mb-3 flex items-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Strategy Success Rankings</span>
            </h3>
            <div className="space-y-3.5">
              {Object.entries(strategy_rankings).map(([strategy, rate]) => (
                <div key={strategy} className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-1.5 font-mono text-[11px]">
                  <div className="flex justify-between font-bold text-slate-200">
                    <span>{strategy}</span>
                    <span className="text-emerald-400">{Math.round(rate * 100)}% Success</span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                    <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${rate * 100}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ================= COLUMN 2: EXECUTION LOGS & REASONING TRACE ================= */}
        <div className="xl:col-span-2 flex flex-col space-y-6">
          {/* Live Traces */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl flex-1 flex flex-col overflow-hidden transition-all hover:border-slate-700">
            <div className="border-b border-slate-800 px-5 py-3.5 bg-slate-900/40 flex items-center justify-between">
              <h2 className="text-xs font-bold flex items-center space-x-2">
                <Terminal className="h-4 w-4 text-amber-400" />
                <span>Live Action Traces & Thoughts</span>
              </h2>
              <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">
                {logs.length} events
              </span>
            </div>

            <div className="flex-1 p-5 overflow-y-auto space-y-4 bg-slate-950/40 font-mono text-xs max-h-[300px]">
              {logs.length > 0 ? (
                logs.map((log, index) => {
                  const isToolSel = log.event_type === "TOOL_SELECTION";
                  return (
                    <div
                      key={index}
                      className={`border-l pl-3 py-1.5 rounded transition-all ${
                        isToolSel ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 hover:bg-slate-900/20'
                      }`}
                    >
                      <div className="flex items-center space-x-2 text-[10px] text-slate-500 mb-0.5">
                        <span>[EVENT]</span>
                        <span className={`font-semibold ${isToolSel ? 'text-indigo-400' : 'text-emerald-400'}`}>
                          {log.event_type}
                        </span>
                      </div>
                      <pre className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">
                        {JSON.stringify(log.details, null, 2)}
                      </pre>
                    </div>
                  );
                })
              ) : (
                <div className="text-center text-slate-500 py-16 flex flex-col items-center justify-center space-y-2">
                  <Sliders className="h-6 w-6 text-slate-600 animate-spin" style={{ animationDuration: '8s' }} />
                  <p className="text-xs">Dispatch instructions to watch live tool routing.</p>
                </div>
              )}
            </div>
          </div>

          {/* Historical Experience Logs */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl transition-all hover:border-slate-700">
            <h3 className="text-xs font-bold text-sky-400 mb-3 flex items-center space-x-2">
              <BookOpen className="h-4 w-4" />
              <span>Historical Experiences & Computational Rewards</span>
            </h3>
            <div className="space-y-3 max-h-[220px] overflow-y-auto pr-1 font-mono text-[11px]">
              {experiences.length > 0 ? (
                experiences.map((exp) => (
                  <div key={exp.id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between">
                    <div>
                      <span className="text-slate-300 font-bold block truncate max-w-[200px]">{exp.objective}</span>
                      <span className="text-[9px] text-slate-500 uppercase">Strategy: {exp.tools_used}</span>
                    </div>
                    <div className="flex items-center space-x-4 text-right">
                      <div>
                        <span className="text-[9px] text-slate-500 block">REWARD</span>
                        <span className={`font-bold ${exp.reward > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {exp.reward > 0 ? '+' : ''}{exp.reward}
                        </span>
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 block">TIME</span>
                        <span className="text-slate-300 font-bold">{exp.execution_time.toFixed(2)}s</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-slate-500 text-xs">No experience records stored in the SQLite experience matrix yet.</p>
              )}
            </div>
          </div>
        </div>

        {/* ================= COLUMN 3: COGNITIVE LOSS GRAPH & TRAJECTORY ================= */}
        <div className="xl:col-span-1 flex flex-col space-y-6">
          {/* Training loss curves metrics list */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col transition-all hover:border-slate-700">
            <h2 className="text-sm font-bold mb-3 flex items-center space-x-2 text-pink-400">
              <TrendingDown className="h-4 w-4" />
              <span>Cognitive Neural Loss History</span>
            </h2>
            <div className="flex-1 overflow-y-auto max-h-[350px] space-y-2 pr-1 font-mono text-[10px]">
              {trainingMetrics.map((m) => (
                <div key={m.epoch} className="bg-slate-950 border border-slate-800/80 p-2.5 rounded-xl flex items-center justify-between">
                  <span className="text-indigo-400 font-bold">EPOCH {m.epoch}</span>
                  <div className="flex items-center space-x-3 text-right">
                    <div>
                      <span className="text-[9px] text-slate-500 block">TRAIN MSE</span>
                      <span className="text-pink-400 font-semibold">{m.train_loss.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-[9px] text-slate-500 block">VAL MSE</span>
                      <span className="text-emerald-400 font-semibold">{m.val_loss.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Repo Structural Index */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl transition-all hover:border-slate-700">
            <h2 className="text-sm font-bold mb-3 flex items-center space-x-2 text-sky-400">
              <Target className="h-4 w-4" />
              <span>Workspace Dependency Analytics</span>
            </h2>
            {repo ? (
              <div className="space-y-3.5 text-xs">
                <div className="grid grid-cols-2 gap-3 font-mono">
                  <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-center">
                    <span className="block text-[9px] text-slate-500">FILES</span>
                    <span className="text-sm font-bold text-sky-400">{repo.total_files}</span>
                  </div>
                  <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-center">
                    <span className="block text-[9px] text-slate-500">PYTHON</span>
                    <span className="text-sm font-bold text-emerald-400">{repo.python_files}</span>
                  </div>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-center text-slate-300 font-mono text-[11px] truncate">
                  {repo.architecture_type}
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">Workspace metrics not calculated.</p>
            )}
          </div>
        </div>

        {/* ================= COLUMN 4: PERSISTENT MEMORY MODULES ================= */}
        <div className="xl:col-span-1 flex flex-col space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col transition-all hover:border-slate-700">
            <h2 className="text-sm font-bold mb-3 flex items-center space-x-2 text-rose-400">
              <Database className="h-4 w-4" />
              <span>Persistent SQLite Memory Heuristics</span>
            </h2>
            <div className="space-y-3 overflow-y-auto max-h-[500px] flex-1 pr-1">
              {memory.length > 0 ? (
                memory.map((item) => (
                  <div key={item.id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl text-[11px] font-mono leading-relaxed transition-all hover:border-slate-700">
                    <div className="flex justify-between text-rose-400 font-semibold mb-1">
                      <span>KEY: {item.key}</span>
                      <span className="text-[9px] text-slate-500 uppercase">{item.category}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed truncate">{item.value}</p>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-600 py-16 text-xs font-mono">
                  No heuristics registered. Run a successful repair to log fix states.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
