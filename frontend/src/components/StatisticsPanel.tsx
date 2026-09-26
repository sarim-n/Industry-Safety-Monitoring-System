import React from 'react';
import { BarChart3, AlertOctagon, Users, Clock } from 'lucide-react';
import type { StatisticsResponse } from '../types/safety';

interface StatisticsPanelProps {
  statistics: StatisticsResponse | null;
  isOnline: boolean;
}

export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({
  statistics,
  isOnline,
}) => {
  const noHelmet = statistics?.no_helmet_count ?? 0;
  const noMask = statistics?.no_mask_count ?? 0;
  const noHelmetMask = statistics?.no_helmet_and_mask_count ?? 0;
  const uniqueWorkers = statistics?.unique_workers ?? 0;
  const latestTs = statistics?.latest_event_timestamp ?? 'None';

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Safety Analytics</h2>
            <p className="text-xs text-slate-400">Aggregate Violation Breakdown</p>
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <AlertOctagon className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-200">NO_HELMET Violations</h4>
              <p className="text-[11px] text-slate-500">Missing safety helmet</p>
            </div>
          </div>
          <span className="text-base font-mono font-bold text-amber-400">
            {isOnline ? noHelmet : '-'}
          </span>
        </div>

        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
              <AlertOctagon className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-200">NO_MASK Violations</h4>
              <p className="text-[11px] text-slate-500">Missing protective mask</p>
            </div>
          </div>
          <span className="text-base font-mono font-bold text-purple-400">
            {isOnline ? noMask : '-'}
          </span>
        </div>

        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
              <AlertOctagon className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-200">NO_HELMET_AND_MASK</h4>
              <p className="text-[11px] text-slate-500">Dual safety violation</p>
            </div>
          </div>
          <span className="text-base font-mono font-bold text-rose-400">
            {isOnline ? noHelmetMask : '-'}
          </span>
        </div>

        <div className="pt-2 grid grid-cols-2 gap-2 text-xs">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-slate-400 mb-1">
              <Users className="w-3.5 h-3.5 text-slate-500" />
              <span>Unique Workers</span>
            </div>
            <p className="font-mono font-bold text-slate-200 text-sm">
              {isOnline ? uniqueWorkers : '-'}
            </p>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-slate-400 mb-1">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>Latest Event</span>
            </div>
            <p className="font-mono font-medium text-slate-300 text-[11px] truncate" title={latestTs}>
              {isOnline ? latestTs : '-'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
