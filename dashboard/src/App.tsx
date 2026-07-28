import React, { useState, useEffect } from 'react';
import { Terminal, Shield, Play, RotateCcw, Cpu, HardDrive, RefreshCw, Layers, Database, Activity, Code } from 'lucide-react';

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

  const fetchDashboardData = async () => {
    try {
      const statusRes = await fetch('/api/status');
      if (statusRes.ok) setStatus(await statusRes.json());

      const logsRes = await fetch('/api/logs');
      if (logsRes.ok) setLogs(await logsRes.json());

      const repoRes = await fetch('/api/repository');
      if (repoRes.ok) setRepo(await repoRes.json());

      const memoryRes = await fetch('/api/memory');
      if (memoryRes.ok) setMemory(await memoryRes.json());

      setApiError(null);
    } catch (err: any) {
      setApiError('Unable to fetch dashboard API. Ensure backend is running.');
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 4000);
    return () => clearInterval(interval);
  }, []);

  const triggerTask = async () => {
    if (!taskInput.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: taskInput }),
      });
      await fetchDashboardData();
    } catch (err) {
      setApiError('Failed to trigger the engineering task execution.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-6 w-6 text-emerald-400 animate-pulse" />
          <h1 className="text-xl font-bold tracking-tight">Autonomous Software Agent Dashboard</h1>
        </div>
        <div className="flex items-center space-x-4">
          {status && (
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              status.status === 'working' ? 'bg-amber-500/20 text-amber-300' : 'bg-emerald-500/20 text-emerald-300'
            }`}>
              {status.status.toUpperCase()}
            </span>
          )}
          <button
            onClick={fetchDashboardData}
            className="p-2 text-slate-400 hover:text-white transition rounded bg-slate-800"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Main Grid View */}
      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-4 gap-6 overflow-auto">
        {/* API Error Notification */}
        {apiError && (
          <div className="xl:col-span-4 bg-red-950/40 border border-red-800 text-red-200 px-4 py-3 rounded-lg flex items-center justify-between">
            <span>{apiError}</span>
            <button onClick={() => setApiError(null)} className="font-semibold underline">Dismiss</button>
          </div>
        )}

        {/* Column 1: Task Execution Console & Telemetry */}
        <div className="xl:col-span-1 space-y-6 flex flex-col">
          {/* Action trigger box */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Play className="h-5 w-5 text-emerald-400" />
              <span>Launch New Task</span>
            </h2>
            <textarea
              className="w-full h-32 bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 mb-4"
              placeholder="Describe the autonomous coding or fixing task (e.g., 'implement divide logic inside src/math.py with tests')"
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
            />
            <button
              onClick={triggerTask}
              disabled={loading || !taskInput.trim()}
              className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 font-semibold text-slate-950 py-2.5 px-4 rounded-lg transition-all flex items-center justify-center space-x-2"
            >
              {loading ? (
                <RefreshCw className="h-5 w-5 animate-spin text-slate-950" />
              ) : (
                <>
                  <Code className="h-5 w-5 text-slate-950" />
                  <span>Execute Autonomously</span>
                </>
              )}
            </button>
          </div>

          {/* System Telemetry */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex-1">
            <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-indigo-400" />
              <span>Sandbox Metrics</span>
            </h2>
            {status ? (
              <div className="space-y-5">
                <div>
                  <div className="flex justify-between text-sm mb-1 text-slate-400">
                    <span>CPU Utilisation</span>
                    <span>{status.system.cpu_usage_pct}%</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-2">
                    <div className="bg-indigo-500 h-2 rounded-full" style={{ width: `${status.system.cpu_usage_pct}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1 text-slate-400">
                    <span>Memory Utilisation</span>
                    <span>{status.system.memory_usage_pct}%</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-2">
                    <div className="bg-pink-500 h-2 rounded-full" style={{ width: `${status.system.memory_usage_pct}%` }}></div>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-800 grid grid-cols-2 gap-4">
                  <div className="bg-slate-950/60 p-3 rounded-lg">
                    <span className="block text-xs text-slate-400">Available RAM</span>
                    <span className="text-lg font-bold text-slate-200">{Math.round(status.system.memory_available_mb)} MB</span>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg">
                    <span className="block text-xs text-slate-400">Self Repair Loops</span>
                    <span className="text-lg font-bold text-slate-200">{status.iteration_count} iterations</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-400">No active system connection.</p>
            )}
          </div>
        </div>

        {/* Column 2: Live Execution Trace Logs */}
        <div className="xl:col-span-2 flex flex-col min-h-[400px]">
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-lg flex-1 flex flex-col overflow-hidden">
            <div className="border-b border-slate-800 px-5 py-4 bg-slate-900/40 flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <Terminal className="h-5 w-5 text-amber-400" />
                <span>Autonomous Trace Log</span>
              </h2>
              <span className="text-xs text-slate-400">{logs.length} Events Logged</span>
            </div>

            <div className="flex-1 p-5 overflow-y-auto space-y-4 bg-slate-950/40 font-mono text-sm max-h-[500px]">
              {logs.length > 0 ? (
                logs.map((log, index) => (
                  <div key={index} className="border-l-2 border-emerald-500/40 pl-4 py-1 hover:bg-slate-900/30 rounded transition-all">
                    <div className="flex items-center space-x-2 text-slate-400 text-xs mb-1">
                      <span>[{log.timestamp}]</span>
                      <span className="text-emerald-400 font-bold">{log.event_type}</span>
                    </div>
                    <pre className="text-xs text-slate-300 whitespace-pre-wrap">
                      {JSON.stringify(log.details, null, 2)}
                    </pre>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-500 py-12">
                  No active engineering traces found. Start a task to observe runtime self-repair traces.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Column 3: Repository & Persistent Memory Insights */}
        <div className="xl:col-span-1 space-y-6 flex flex-col">
          {/* Repository overview */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Layers className="h-5 w-5 text-sky-400" />
              <span>Project Structure</span>
            </h2>
            {repo ? (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/40">
                  <span className="block text-xs text-slate-400">Total Files</span>
                  <span className="text-lg font-bold text-sky-400">{repo.total_files}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/40">
                  <span className="block text-xs text-slate-400">Python Files</span>
                  <span className="text-lg font-bold text-emerald-400">{repo.python_files}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/40 col-span-2">
                  <span className="block text-xs text-slate-400">Architecture Class</span>
                  <span className="text-sm font-semibold text-slate-200 mt-1">{repo.architecture_type}</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-400">No project structure scanned.</p>
            )}
          </div>

          {/* SQLite persistent memory matches */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex-1">
            <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Database className="h-5 w-5 text-rose-400" />
              <span>Persistent Memory (SQLite)</span>
            </h2>
            <div className="space-y-3 overflow-y-auto max-h-[250px]">
              {memory.length > 0 ? (
                memory.map((item) => (
                  <div key={item.id} className="bg-slate-950 border border-slate-800/60 p-3 rounded-lg text-xs space-y-1">
                    <div className="flex justify-between text-rose-400 font-semibold mb-1">
                      <span>Key: {item.key}</span>
                      <span className="text-[10px] text-slate-400 font-normal uppercase">{item.category}</span>
                    </div>
                    <p className="text-slate-300 font-mono">{item.value}</p>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-400">No matching persistent memories retrieved yet.</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
