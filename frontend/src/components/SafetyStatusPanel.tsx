import React from 'react';
import { ShieldCheck, AlertOctagon, User, HelpCircle } from 'lucide-react';
import type { WorkerStateModel, SafetyStatus } from '../types/safety';

interface SafetyStatusPanelProps {
  workers: WorkerStateModel[];
  isOnline: boolean;
}

export const getStatusBadgeStyle = (status: SafetyStatus | string) => {
  switch (status) {
    case 'SAFE':
      return {
        label: 'SAFE',
        bg: 'bg-emerald-500/10',
        text: 'text-emerald-400',
        border: 'border-emerald-500/30',
        icon: ShieldCheck,
      };
    case 'NO_HELMET':
      return {
        label: 'NO HELMET',
        bg: 'bg-amber-500/10',
        text: 'text-amber-400',
        border: 'border-amber-500/30',
        icon: AlertOctagon,
      };
    case 'NO_MASK':
      return {
        label: 'NO MASK',
        bg: 'bg-purple-500/10',
        text: 'text-purple-400',
        border: 'border-purple-500/30',
        icon: AlertOctagon,
      };
    case 'NO_HELMET_AND_MASK':
      return {
        label: 'NO HELMET & MASK',
        bg: 'bg-rose-500/10',
        text: 'text-rose-400 font-bold',
        border: 'border-rose-500/40',
        icon: AlertOctagon,
      };
    case 'UNCERTAIN':
    default:
      return {
        label: 'UNCERTAIN',
        bg: 'bg-cyan-500/10',
        text: 'text-cyan-400',
        border: 'border-cyan-500/30',
        icon: HelpCircle,
      };
  }
};

export const SafetyStatusPanel: React.FC<SafetyStatusPanelProps> = ({
  workers,
  isOnline,
}) => {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <User className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Live Worker Safety Status</h2>
            <p className="text-xs text-slate-400">Tracked Worker PPE Compliance</p>
          </div>
        </div>

        <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-mono">
          Active: {isOnline ? workers.length : 0}
        </span>
      </div>

      <div className="mt-4 space-y-2.5 overflow-y-auto max-h-[380px] pr-1">
        {!isOnline ? (
          <div className="p-6 text-center text-slate-500 border border-dashed border-slate-800 rounded-lg text-xs">
            Backend server offline. Connect FastAPI to view worker states.
          </div>
        ) : workers.length === 0 ? (
          <div className="p-6 text-center text-slate-400 border border-dashed border-slate-800 rounded-lg text-xs">
            No active workers currently detected in CV frame.
          </div>
        ) : (
          workers.map((worker) => {
            const style = getStatusBadgeStyle(worker.status);
            const Icon = style.icon;

            return (
              <div
                key={worker.worker_id}
                className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 flex items-center justify-between transition-all hover:border-slate-700"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-mono text-xs text-slate-300 font-bold">
                    #{worker.worker_id}
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-slate-200">
                      Worker #{worker.worker_id}
                    </h4>
                    <p className="text-[11px] text-slate-500 font-mono">
                      Track ID: #{worker.worker_id}
                    </p>
                  </div>
                </div>

                <div
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md border text-xs font-semibold ${style.bg} ${style.text} ${style.border}`}
                >
                  <Icon className="w-3.5 h-3.5 shrink-0" />
                  <span>{style.label}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
