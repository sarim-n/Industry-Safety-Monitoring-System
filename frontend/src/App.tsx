import React, { useState } from 'react';
import { DashboardPage } from './pages/DashboardPage';
import { VideoAnalysisPage } from './pages/VideoAnalysisPage';

export function App() {
  const [currentView, setCurrentView] = useState<'dashboard' | 'video'>('dashboard');

  return (
    <div className="flex flex-col min-h-screen bg-[#0b0f19]">
      {/* Super simple global nav bar placed above the pages */}
      <div className="bg-slate-950 border-b border-slate-800/80 px-6 py-3 flex items-center justify-center gap-4">
        <button
          onClick={() => setCurrentView('dashboard')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            currentView === 'dashboard'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'text-slate-400 hover:bg-slate-800'
          }`}
        >
          Live Dashboard
        </button>
        <button
          onClick={() => setCurrentView('video')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            currentView === 'video'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:bg-slate-800'
          }`}
        >
          Video Upload Analysis
        </button>
      </div>

      <div className="flex-1 overflow-auto">
        {currentView === 'dashboard' ? <DashboardPage /> : <VideoAnalysisPage />}
      </div>
    </div>
  );
}

export default App;
