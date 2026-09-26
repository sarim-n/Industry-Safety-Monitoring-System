import React, { useState } from 'react';
import { X, ImageOff, ExternalLink, Calendar, User, ShieldAlert } from 'lucide-react';
import type { EventModel } from '../types/safety';
import { safetyApi } from '../services/api';
import { getStatusBadgeStyle } from './SafetyStatusPanel';

interface EvidenceModalProps {
  event: EventModel | null;
  onClose: () => void;
}

export const EvidenceModal: React.FC<EvidenceModalProps> = ({ event, onClose }) => {
  const [imageError, setImageError] = useState(false);

  if (!event) return null;

  const imageUrl = safetyApi.getEvidenceUrl(event.evidence_path);
  const badgeStyle = getStatusBadgeStyle(event.violation_type);

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full overflow-hidden shadow-2xl transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-slate-950 px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/30">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">Evidence Snapshot Viewer</h3>
              <p className="text-xs text-slate-400 font-mono">
                Worker #{event.worker_id} — {event.violation_type}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-100 transition-colors border border-slate-700 cursor-pointer"
            title="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="bg-slate-950 p-6 flex items-center justify-center min-h-[320px] max-h-[500px] relative">
          {imageError ? (
            <div className="text-center p-8 text-slate-400 max-w-sm">
              <ImageOff className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <h4 className="text-sm font-semibold text-slate-300">Evidence Image Unavailable</h4>
              <p className="text-xs text-slate-500 mt-1">
                The evidence file could not be loaded from the backend or was moved.
              </p>
              <code className="text-[11px] font-mono bg-slate-900 px-2.5 py-1 rounded text-slate-400 border border-slate-800 block mt-3 break-all">
                {event.evidence_path}
              </code>
            </div>
          ) : (
            <img
              src={imageUrl}
              alt={`Evidence for Worker #${event.worker_id} ${event.violation_type}`}
              onError={() => setImageError(true)}
              className="max-h-[460px] w-auto object-contain rounded-lg border border-slate-800 shadow-lg"
            />
          )}
        </div>

        <div className="p-6 bg-slate-900 border-t border-slate-800 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <div className="flex items-center gap-1.5 text-slate-400 mb-1">
              <Calendar className="w-3.5 h-3.5 text-slate-500" />
              <span>Timestamp</span>
            </div>
            <p className="font-mono font-semibold text-slate-200">{event.timestamp}</p>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <div className="flex items-center gap-1.5 text-slate-400 mb-1">
              <User className="w-3.5 h-3.5 text-slate-500" />
              <span>Target Worker</span>
            </div>
            <p className="font-mono font-semibold text-slate-200">Worker #{event.worker_id}</p>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <div className="flex items-center gap-1.5 text-slate-400 mb-1">
              <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
              <span>Confirmed Violation</span>
            </div>
            <span
              className={`inline-block px-2 py-0.5 rounded border text-[11px] font-semibold ${badgeStyle.bg} ${badgeStyle.text} ${badgeStyle.border}`}
            >
              {badgeStyle.label}
            </span>
          </div>
        </div>

        <div className="px-6 py-3 bg-slate-950 border-t border-slate-800 flex justify-between items-center text-xs">
          <a
            href={imageUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-400 hover:text-cyan-400 transition-colors inline-flex items-center gap-1 font-medium"
          >
            <span>Open raw image in new tab</span>
            <ExternalLink className="w-3 h-3" />
          </a>

          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 font-medium transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
