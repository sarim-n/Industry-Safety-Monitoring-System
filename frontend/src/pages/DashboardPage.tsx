import React, { useState } from 'react';
import { useSafetyData } from '../hooks/useSafetyData';
import { Header } from '../components/Header';
import { SummaryCards } from '../components/SummaryCards';
import { LiveMonitorPanel } from '../components/LiveMonitorPanel';
import { SafetyStatusPanel } from '../components/SafetyStatusPanel';
import { RecentEventsPanel } from '../components/RecentEventsPanel';
import { StatisticsPanel } from '../components/StatisticsPanel';
import { EvidenceModal } from '../components/EvidenceModal';
import type { EventModel } from '../types/safety';

export const DashboardPage: React.FC = () => {
  const {
    status,
    workers,
    events,
    statistics,
    isOnline,
    lastUpdated,
    refreshNow,
  } = useSafetyData(2500);

  const [selectedEvidence, setSelectedEvidence] = useState<EventModel | null>(null);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans">
      <Header
        isOnline={isOnline}
        status={status}
        lastUpdated={lastUpdated}
        onRefresh={refreshNow}
      />

      <main className="flex-1 max-w-[1920px] w-full mx-auto p-4 md:p-6 space-y-6">
        <SummaryCards
          workers={workers}
          statistics={statistics}
          isOnline={isOnline}
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <LiveMonitorPanel status={status} isOnline={isOnline} />
          </div>
          <div className="lg:col-span-1">
            <SafetyStatusPanel workers={workers} isOnline={isOnline} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <RecentEventsPanel
              events={events}
              isOnline={isOnline}
              onSelectEvidence={(evt) => setSelectedEvidence(evt)}
            />
          </div>
          <div className="lg:col-span-1">
            <StatisticsPanel statistics={statistics} isOnline={isOnline} />
          </div>
        </div>
      </main>

      <footer className="bg-slate-950 border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500">
        Industrial Safety Monitoring System &copy; 2026 — Phase 6 Control Room Dashboard
      </footer>

      {selectedEvidence && (
        <EvidenceModal
          event={selectedEvidence}
          onClose={() => setSelectedEvidence(null)}
        />
      )}
    </div>
  );
};
