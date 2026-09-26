import React from 'react';
import { Video, Cpu, ShieldAlert, Layers, Terminal } from 'lucide-react';
import type { SystemStatusResponse } from '../types/safety';

interface LiveMonitorPanelProps {
  status: SystemStatusResponse | null;
  isOnline: boolean;
}

export const LiveMonitorPanel: React.FC<LiveMonitorPanelProps> = ({
  status,
  isOnline,
}) => {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Video className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Live Video Monitor</h2>
            <p className="text-xs text-slate-400">Computer Vision Stream Telemetry</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Resolution: 800x800
          </span>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-emerald-400 border border-slate-700">
            YOLOv8s @ 800
          </span>
        </div>
      </div>

      <div className="my-5 flex-1 bg-slate-950 rounded-lg border border-slate-800 p-6 flex flex-col items-center justify-center relative overflow-hidden group">
        <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:16px_16px] opacity-40 pointer-events-none"></div>

        <div className="relative z-10 text-center max-w-md mx-auto">
          <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center mx-auto mb-4 text-amber-400 shadow-inner">
            <Cpu className="w-8 h-8 animate-pulse" />
          </div>

          <h3 className="text-lg font-bold text-slate-100 mb-1">
            Live Stream Operating in CV Pipeline
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            OpenCV video processing loop executes independently in Python with GPU acceleration. Real-time bounding boxes and safety telemetry are synced via REST bridge.
          </p>

          <div className="bg-slate-900/90 rounded-md p-3 border border-slate-800 text-left font-mono text-xs text-slate-300 flex items-start gap-2">
            <Terminal className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <code className="break-all text-slate-200">
              python run_live.py --source &lt;video&gt; --enable-api
            </code>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Layers className="w-3.5 h-3.5 text-slate-500" />
            <span>CONFIDENCE THRESHOLDS</span>
          </div>
          <p className="text-xs font-mono font-semibold text-slate-200">
            Person: 0.50 | Helmet: 0.25 | Mask: 0.20
          </p>
        </div>

        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
            <span>TEMPORAL STREAK</span>
          </div>
          <p className="text-xs font-mono font-semibold text-slate-200">
            5 Consecutive Frames (5.0s Cooldown)
          </p>
        </div>

        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Cpu className="w-3.5 h-3.5 text-slate-500" />
            <span>PIPELINE ENGINE</span>
          </div>
          <p className="text-xs font-mono font-semibold text-emerald-400">
            PPE Associator + Temporal Confirmation
          </p>
        </div>

        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
            <Video className="w-3.5 h-3.5 text-slate-500" />
            <span>STREAM STATE</span>
          </div>
          <p className="text-xs font-mono font-semibold text-slate-200">
            {isOnline && status?.system_running ? (
              <span className="text-emerald-400">Active & Syncing</span>
            ) : (
              <span className="text-slate-400">Standby (Backend Ready)</span>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};
