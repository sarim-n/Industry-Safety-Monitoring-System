import React from 'react';
import { Users, ShieldCheck, AlertTriangle, FileText } from 'lucide-react';
import type { WorkerStateModel, StatisticsResponse } from '../types/safety';

interface SummaryCardsProps {
  workers: WorkerStateModel[];
  statistics: StatisticsResponse | null;
  isOnline: boolean;
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({
  workers,
  statistics,
  isOnline,
}) => {
  const activeWorkersCount = workers.length;
  const safeWorkersCount = workers.filter((w) => w.status === 'SAFE').length;
  const activeViolationsCount = workers.filter(
    (w) =>
      w.status === 'NO_HELMET' ||
      w.status === 'NO_MASK' ||
      w.status === 'NO_HELMET_AND_MASK'
  ).length;

  const totalEventsCount = statistics?.total_events ?? 0;

  const cards = [
    {
      title: 'Active Workers',
      value: isOnline ? activeWorkersCount : '-',
      subtitle: 'Currently tracked on floor',
      icon: Users,
      color: 'text-cyan-400',
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/20',
    },
    {
      title: 'Safe Workers',
      value: isOnline ? safeWorkersCount : '-',
      subtitle: 'Compliant with helmet & mask',
      icon: ShieldCheck,
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
    },
    {
      title: 'Active Violations',
      value: isOnline ? activeViolationsCount : '-',
      subtitle: 'Active non-compliance tracks',
      icon: AlertTriangle,
      color: activeViolationsCount > 0 ? 'text-rose-400' : 'text-slate-400',
      bg: activeViolationsCount > 0 ? 'bg-rose-500/10' : 'bg-slate-800/40',
      border: activeViolationsCount > 0 ? 'border-rose-500/30' : 'border-slate-800',
    },
    {
      title: 'Total Events Logged',
      value: isOnline ? totalEventsCount : '-',
      subtitle: 'Recorded in events.csv',
      icon: FileText,
      color: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className={`bg-slate-900 rounded-xl border ${card.border} p-5 flex items-start justify-between shadow-sm transition-all hover:border-slate-700`}
          >
            <div>
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                {card.title}
              </p>
              <h3 className={`text-3xl font-extrabold mt-2 ${card.color}`}>
                {card.value}
              </h3>
              <p className="text-xs text-slate-500 mt-1">{card.subtitle}</p>
            </div>
            <div className={`p-3 rounded-lg ${card.bg} border ${card.border}`}>
              <Icon className={`w-5 h-5 ${card.color}`} />
            </div>
          </div>
        );
      })}
    </div>
  );
};
