import React from 'react';
import { History, Image, ChevronRight, FileText } from 'lucide-react';
import type { EventModel } from '../types/safety';
import { getStatusBadgeStyle } from './SafetyStatusPanel';

interface RecentEventsPanelProps {
  events: EventModel[];
  isOnline: boolean;
  onSelectEvidence: (event: EventModel) => void;
}

export const RecentEventsPanel: React.FC<RecentEventsPanelProps> = ({
  events,
  isOnline,
  onSelectEvidence,
}) => {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col shadow-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-md bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <History className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Confirmed Safety Events</h2>
            <p className="text-xs text-slate-400">Recorded Violation Log (events.csv)</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
          <FileText className="w-3.5 h-3.5 text-slate-500" />
          <span>Total Recorded: {events.length}</span>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider bg-slate-950/50">
              <th className="py-3 px-4">Timestamp</th>
              <th className="py-3 px-4">Worker ID</th>
              <th className="py-3 px-4">Violation Type</th>
              <th className="py-3 px-4">Evidence</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {!isOnline ? (
              <tr>
                <td colSpan={5} className="py-8 text-center text-slate-500 font-sans">
                  Backend offline. Cannot load events.
                </td>
              </tr>
            ) : events.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-8 text-center text-slate-400 font-sans">
                  No safety violation events logged yet.
                </td>
              </tr>
            ) : (
              events.map((evt, idx) => {
                const badgeStyle = getStatusBadgeStyle(evt.violation_type);
                const hasEvidence = Boolean(evt.evidence_path);

                return (
                  <tr
                    key={idx}
                    className="hover:bg-slate-800/40 transition-colors group"
                  >
                    <td className="py-3.5 px-4 text-slate-300 font-medium whitespace-nowrap">
                      {evt.timestamp}
                    </td>

                    <td className="py-3.5 px-4 text-slate-200">
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        Worker #{evt.worker_id}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded border text-[11px] font-semibold ${badgeStyle.bg} ${badgeStyle.text} ${badgeStyle.border}`}
                      >
                        {badgeStyle.label}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 whitespace-nowrap">
                      {hasEvidence ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400 text-[11px]">
                          <Image className="w-3.5 h-3.5" />
                          <span>JPEG Available</span>
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[11px]">None</span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 text-right whitespace-nowrap">
                      {hasEvidence ? (
                        <button
                          onClick={() => onSelectEvidence(evt)}
                          className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 transition-colors inline-flex items-center gap-1 text-xs font-sans font-medium cursor-pointer"
                        >
                          <span>View Evidence</span>
                          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                        </button>
                      ) : (
                        <span className="text-slate-600 text-xs">-</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
