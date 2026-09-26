import React from 'react';
import { ShieldAlert, RefreshCw, Activity, ServerOff } from 'lucide-react';
import type { SystemStatusResponse } from '../types/safety';

interface HeaderProps {
  isOnline: boolean;
  status: SystemStatusResponse | null;
  lastUpdated: Date | null;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  isOnline,
  status,
  lastUpdated,
  onRefresh,
}) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 sticky top-0 z-30 shadow-md">
      <div className="max-w-[1920px] mx-auto flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
              Industrial Safety Monitor
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                v1.0.0 (Phase 6)
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              YOLOv8 Real-Time PPE & Worker Violation System
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-xs font-medium bg-slate-950">
            {isOnline ? (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-semibold tracking-wide">SYSTEM ONLINE</span>
                {status?.system_running && (
                  <span className="text-slate-500 border-l border-slate-800 pl-2">
                    CV PIPELINE ACTIVE
                  </span>
                )}
              </>
            ) : (
              <>
                <ServerOff className="w-3.5 h-3.5 text-rose-500" />
                <span className="text-rose-400 font-semibold tracking-wide">BACKEND OFFLINE</span>
              </>
            )}
          </div>

          {lastUpdated && (
            <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-400 bg-slate-950 px-3 py-1.5 rounded-md border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              <span>
                Updated: <strong className="text-slate-200">{lastUpdated.toLocaleTimeString()}</strong>
              </span>
            </div>
          )}

          <button
            onClick={onRefresh}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-colors flex items-center gap-1.5 text-xs font-medium cursor-pointer"
            title="Poll API immediately"
          >
            <RefreshCw className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden sm:inline">Refresh</span>
          </button>
        </div>
      </div>
    </header>
  );
};
