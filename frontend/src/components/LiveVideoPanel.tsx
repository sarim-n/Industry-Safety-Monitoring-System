import React, { useState } from 'react';
import { Video, VideoOff, Layers, ShieldAlert, Cpu } from 'lucide-react';
import type { SystemStatusResponse } from '../types/safety';
import { safetyApi } from '../services/api';

interface LiveVideoPanelProps {
  status: SystemStatusResponse | null;
  isOnline: boolean;
}

export const LiveVideoPanel: React.FC<LiveVideoPanelProps> = ({
  status,
  isOnline,
}) => {
  const [streamError, setStreamError] = useState(false);
  const streamUrl = safetyApi.getVideoStreamUrl();

  const isLive = isOnline && status?.system_running && !streamError;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full shadow-sm">
      {/* Panel Header */}
      <div className="flex flex-wrap items-center justify-between pb-4 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Video className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Live Safety Monitor Feed</h2>
            <p className="text-xs text-slate-400">OpenCV + YOLOv8s @ 800 Annotated Stream</p>
          </div>
        </div>

        {/* Live Indicator Badges */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-medium bg-slate-950">
            {isLive ? (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-semibold tracking-wide">LIVE CAMERA FEED</span>
              </>
            ) : (
              <>
                <VideoOff className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-amber-400 font-semibold tracking-wide">STREAM STANDBY / OFFLINE</span>
              </>
            )}
          </div>

          <span className="hidden sm:inline-block text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-emerald-400 border border-slate-700">
            YOLOv8s @ 800
          </span>
        </div>
      </div>

      {/* Main Video Screen Container */}
      <div className="my-4 flex-1 bg-slate-950 rounded-lg border border-slate-800 flex items-center justify-center relative overflow-hidden min-h-[360px]">
        {/* MJPEG Stream Image */}
        <img
          src={streamUrl}
          alt="Live Industrial AI Safety Video Stream"
          onError={() => setStreamError(true)}
          onLoad={() => setStreamError(false)}
          className="w-full h-full object-contain rounded-lg max-h-[520px]"
        />
      </div>

      {/* Pipeline Specs Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1">
        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] mb-0.5">
            <Layers className="w-3 h-3 text-slate-500" />
            <span>DETECTION THRESHOLDS</span>
          </div>
          <p className="text-xs font-mono font-semibold text-slate-200">
            Person: 0.50 | Helmet: 0.25 | Mask: 0.20
          </p>
        </div>

        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] mb-0.5">
            <ShieldAlert className="w-3 h-3 text-slate-500" />
            <span>TEMPORAL STREAK</span>
          </div>
          <p className="text-xs font-mono font-semibold text-slate-200">
            5 Frames (5.0s Cooldown)
          </p>
        </div>

        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] mb-0.5">
            <Cpu className="w-3 h-3 text-slate-500" />
            <span>AI ENGINES</span>
          </div>
          <p className="text-xs font-mono font-semibold text-emerald-400">
            PPE Associator + Track Filter
          </p>
        </div>

        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] mb-0.5">
            <Video className="w-3 h-3 text-slate-500" />
            <span>STREAM FORMAT</span>
          </div>
          <p className="text-xs font-mono font-semibold text-cyan-400">
            FastAPI MJPEG (800x800)
          </p>
        </div>
      </div>
    </div>
  );
};
