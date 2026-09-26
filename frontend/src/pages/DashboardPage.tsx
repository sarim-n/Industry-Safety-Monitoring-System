import React, { useState } from 'react';
import { useSafetyData } from '../hooks/useSafetyData';
import { Header } from '../components/Header';
import { SummaryCards } from '../components/SummaryCards';
import { LiveVideoPanel } from '../components/LiveVideoPanel';
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
        {/* Row 1: Summary Cards */}
        <SummaryCards
          workers={workers}
          statistics={statistics}
          isOnline={isOnline}
        />

        {/* Row 2: Live Processed Video Stream & Live Worker Safety Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <LiveVideoPanel status={status} isOnline={isOnline} />
          </div>
          <div className="lg:col-span-1">
            <SafetyStatusPanel workers={workers} isOnline={isOnline} />
          </div>
        </div>

        {/* Row 3: Recent Events Table & Statistics */}
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
        Industrial Safety Monitoring System &copy; 2026 — Phase 7 Live Video Dashboard
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
