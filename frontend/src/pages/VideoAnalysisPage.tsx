import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload, Film, CheckCircle2, AlertTriangle, Loader2,
  HardHat, Smile, ShieldOff, Users, Zap, RefreshCw, X,
  PlayCircle, ChevronRight, FileVideo,
} from 'lucide-react';
import type { VideoStatusResponse, VideoResultsResponse } from '../types/video';

const API_BASE = 'http://127.0.0.1:8000';
const POLL_INTERVAL_MS = 1200;
const ALLOWED_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'];

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; label: string }> = {
    queued:     { color: 'bg-slate-700 text-slate-300',       label: 'Queued' },
    processing: { color: 'bg-amber-500/20 text-amber-300',   label: 'Processing…' },
    completed:  { color: 'bg-emerald-500/20 text-emerald-300', label: 'Completed' },
    failed:     { color: 'bg-rose-500/20 text-rose-300',      label: 'Failed' },
  };
  const { color, label } = map[status] ?? { color: 'bg-slate-700 text-slate-400', label: status };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold ${color}`}>
      {status === 'processing' && <Loader2 className="w-3 h-3 animate-spin" />}
      {status === 'completed' && <CheckCircle2 className="w-3 h-3" />}
      {status === 'failed' && <AlertTriangle className="w-3 h-3" />}
      {label}
    </span>
  );
}

function StatCard({
  icon, label, value, color,
}: {
  icon: React.ReactNode; label: string; value: number | string; color: string;
}) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4 flex items-center gap-3">
      <div className={`p-2.5 rounded-lg ${color}`}>{icon}</div>
      <div>
        <p className="text-xs text-slate-400 font-medium">{label}</p>
        <p className="text-xl font-bold text-slate-100">{value}</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────────────────
export const VideoAnalysisPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<VideoStatusResponse | null>(null);
  const [results, setResults] = useState<VideoResultsResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // ── File selection ──────────────────────────────────────────────────────
  const handleFile = (f: File) => {
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      setUploadError(`Unsupported file type "${ext}". Allowed: ${ALLOWED_EXTS.join(', ')}`);
      return;
    }
    setFile(f);
    setUploadError(null);
    setJobId(null);
    setJobStatus(null);
    setResults(null);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  // ── Upload & Start Processing ───────────────────────────────────────────
  const startAnalysis = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setJobId(null);
    setJobStatus(null);
    setResults(null);

    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE}/api/video/upload`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? `Upload failed (${res.status})`);
      }
      const data = await res.json();
      setJobId(data.job_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload error';
      setUploadError(msg);
    } finally {
      setUploading(false);
    }
  };

  // ── Poll Status ─────────────────────────────────────────────────────────
  const pollStatus = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/video/status/${id}`);
      if (!res.ok) return;
      const data: VideoStatusResponse = await res.json();
      setJobStatus(data);

      if (data.status === 'completed') {
        // Fetch results
        const rRes = await fetch(`${API_BASE}/api/video/results/${id}`);
        if (rRes.ok) {
          const rData: VideoResultsResponse = await rRes.json();
          setResults(rData);
        }
        return; // stop polling
      }
      if (data.status === 'failed') return; // stop polling

      // Continue polling
      pollRef.current = setTimeout(() => pollStatus(id), POLL_INTERVAL_MS);
    } catch {
      // network error — retry
      pollRef.current = setTimeout(() => pollStatus(id), POLL_INTERVAL_MS * 2);
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setTimeout(() => pollStatus(jobId), 500);
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [jobId, pollStatus]);

  // ── Reset ───────────────────────────────────────────────────────────────
  const reset = () => {
    if (pollRef.current) clearTimeout(pollRef.current);
    setFile(null);
    setJobId(null);
    setJobStatus(null);
    setResults(null);
    setUploadError(null);
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const videoUrl = jobStatus?.status === 'completed' && jobId
    ? `${API_BASE}/api/video/output/${jobId}`
    : null;

  const progressPct = jobStatus?.progress ?? 0;
  const isProcessing = jobStatus?.status === 'processing' || jobStatus?.status === 'queued';

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans">
      {/* ── Header ── */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 sticky top-0 z-30 shadow-md">
        <div className="max-w-5xl mx-auto flex items-center gap-3">
          <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400">
            <Film className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100 tracking-tight">
              Video Safety Analysis
              <span className="ml-2 text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                YOLOv8s Run 2B
              </span>
            </h1>
            <p className="text-xs text-slate-400">Upload a video — process through the full production PPE safety pipeline</p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto p-4 md:p-8 space-y-8">

        {/* ── Upload Card ── */}
        {!jobId && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Upload className="w-4 h-4 text-amber-400" />
              Upload Video for Analysis
            </h2>

            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative flex flex-col items-center justify-center gap-4 cursor-pointer
                rounded-xl border-2 border-dashed transition-all duration-200 p-10
                ${dragOver
                  ? 'border-amber-400 bg-amber-400/5'
                  : file
                    ? 'border-emerald-600/60 bg-emerald-900/10'
                    : 'border-slate-700 hover:border-slate-500 bg-slate-800/30 hover:bg-slate-800/50'}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={onInputChange}
              />
              {file ? (
                <>
                  <FileVideo className="w-12 h-12 text-emerald-400" />
                  <div className="text-center">
                    <p className="font-semibold text-slate-100">{file.name}</p>
                    <p className="text-sm text-slate-400 mt-1">{formatBytes(file.size)}</p>
                  </div>
                  <p className="text-xs text-slate-500">Click or drop to change</p>
                </>
              ) : (
                <>
                  <div className="p-4 rounded-full bg-slate-800 border border-slate-700">
                    <Upload className="w-8 h-8 text-slate-400" />
                  </div>
                  <div className="text-center">
                    <p className="font-medium text-slate-300">Drop a video here, or click to browse</p>
                    <p className="text-sm text-slate-500 mt-1">
                      Supported: {ALLOWED_EXTS.join(', ')}
                    </p>
                  </div>
                </>
              )}
            </div>

            {/* Upload Error */}
            {uploadError && (
              <div className="mt-4 flex items-center gap-2 text-rose-400 bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 text-sm">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Start button */}
            {file && !uploading && (
              <button
                onClick={startAnalysis}
                className="mt-5 w-full flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold py-3 px-6 rounded-xl transition-colors text-sm cursor-pointer"
              >
                <PlayCircle className="w-4 h-4" />
                Start Safety Analysis
              </button>
            )}

            {uploading && (
              <div className="mt-5 flex items-center justify-center gap-2 text-slate-300 text-sm">
                <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                Uploading and starting analysis…
              </div>
            )}
          </div>
        )}

        {/* ── Progress Card (while processing) ── */}
        {jobId && isProcessing && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                Processing Video
              </h2>
              <StatusBadge status={jobStatus?.status ?? 'queued'} />
            </div>

            {/* Progress bar */}
            <div className="w-full bg-slate-700/50 rounded-full h-3 overflow-hidden mb-3">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-400">
              <span>
                {jobStatus?.total_frames
                  ? `Frame ${jobStatus.current_frame.toLocaleString()} / ${jobStatus.total_frames.toLocaleString()}`
                  : `Frame ${jobStatus?.current_frame?.toLocaleString() ?? 0}`}
              </span>
              <span className="font-semibold text-amber-400">{progressPct.toFixed(1)}%</span>
            </div>

            <p className="text-xs text-slate-500 mt-4">
              Running through YOLOv8s Run 2B → Person Suppression → PPE Association → Temporal Confirmation…
            </p>
          </div>
        )}

        {/* ── Failed Card ── */}
        {jobStatus?.status === 'failed' && (
          <div className="bg-rose-900/20 border border-rose-700/50 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center gap-3 mb-3">
              <AlertTriangle className="w-6 h-6 text-rose-400" />
              <h2 className="text-base font-semibold text-rose-300">Processing Failed</h2>
            </div>
            <p className="text-sm text-rose-400">{jobStatus.error ?? 'Unknown error.'}</p>
            <button
              onClick={reset}
              className="mt-4 flex items-center gap-2 text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-lg border border-slate-700 cursor-pointer transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Try Again
            </button>
          </div>
        )}

        {/* ── Completed: Video + Results ── */}
        {jobStatus?.status === 'completed' && videoUrl && (
          <>
            {/* Video Player */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-800">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  Annotated Output Video — Bounding Boxes Burned In
                </h2>
                <StatusBadge status="completed" />
              </div>
              <div className="bg-black">
                <video
                  ref={videoRef}
                  key={videoUrl}
                  controls
                  className="w-full max-h-[60vh] outline-none"
                  style={{ background: '#000' }}
                  onError={() => {
                    // If browser can't play inline, the download link below will still work
                    console.warn('Browser video playback error — use download link for VLC');
                  }}
                >
                  {/* Try both types for codec fallback */}
                  <source src={videoUrl} type="video/mp4" />
                  <source src={videoUrl} type="video/x-msvideo" />
                  Your browser does not support HTML5 video.
                </video>
              </div>
              <div className="px-5 py-3 bg-slate-950/50 text-xs text-slate-500 flex flex-wrap gap-x-4 gap-y-1 items-center">
                <span>Job: <code className="text-slate-400">{jobId?.slice(0, 8)}…</code></span>
                {results && (
                  <>
                    <span>Codec: <span className="text-slate-300">{results.codec_used}</span></span>
                    <span>Model: <span className="text-slate-300">{results.model.split('—')[0].trim()}</span></span>
                  </>
                )}
                <a
                  href={videoUrl}
                  download
                  className="ml-auto flex items-center gap-1.5 text-amber-400 hover:text-amber-300 transition-colors font-medium"
                >
                  ↓ Download annotated video
                </a>
              </div>
            </div>


            {/* Results Grid */}
            {results && (
              <div className="space-y-4">
                <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  Analysis Results
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatCard
                    icon={<Film className="w-4 h-4 text-sky-400" />}
                    label="Frames Processed"
                    value={results.frames_processed.toLocaleString()}
                    color="bg-sky-500/10 border border-sky-500/20"
                  />
                  <StatCard
                    icon={<Zap className="w-4 h-4 text-amber-400" />}
                    label="Processing FPS"
                    value={results.processing_fps.toFixed(1)}
                    color="bg-amber-500/10 border border-amber-500/20"
                  />
                  <StatCard
                    icon={<Users className="w-4 h-4 text-violet-400" />}
                    label="Workers Detected"
                    value={results.workers_detected}
                    color="bg-violet-500/10 border border-violet-500/20"
                  />
                  <StatCard
                    icon={<ShieldOff className="w-4 h-4 text-rose-400" />}
                    label="Confirmed Violations"
                    value={results.confirmed_violations}
                    color="bg-rose-500/10 border border-rose-500/20"
                  />
                </div>

                {/* Violation breakdown */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-300">Violation Breakdown</h3>
                  {[
                    { label: 'No Helmet', value: results.no_helmet, color: 'bg-orange-500', icon: <HardHat className="w-4 h-4 text-orange-400" /> },
                    { label: 'No Mask', value: results.no_mask, color: 'bg-fuchsia-500', icon: <Smile className="w-4 h-4 text-fuchsia-400" /> },
                    { label: 'No Helmet & Mask', value: results.no_helmet_and_mask, color: 'bg-rose-600', icon: <ShieldOff className="w-4 h-4 text-rose-400" /> },
                  ].map(({ label, value, color, icon }) => (
                    <div key={label} className="flex items-center gap-3">
                      {icon}
                      <span className="text-sm text-slate-300 w-36">{label}</span>
                      <div className="flex-1 bg-slate-700/50 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-2 rounded-full ${color} transition-all duration-500`}
                          style={{
                            width: results.confirmed_violations > 0
                              ? `${(value / results.confirmed_violations) * 100}%`
                              : '0%',
                          }}
                        />
                      </div>
                      <span className="text-sm font-bold text-slate-200 w-6 text-right">{value}</span>
                    </div>
                  ))}
                  <div className="pt-2 border-t border-slate-800 text-xs text-slate-500 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    {results.evidence_count} evidence image{results.evidence_count !== 1 ? 's' : ''} saved to <code className="text-slate-400">evidence/</code>
                  </div>
                </div>

                {/* New analysis button */}
                <button
                  onClick={reset}
                  className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 px-5 py-2.5 rounded-xl border border-slate-700 text-sm font-medium cursor-pointer transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Process Another Video
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                </button>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="bg-slate-950 border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500">
        Industrial Safety Monitoring System © 2026 — Video Analysis · YOLOv8s Run 2B
      </footer>
    </div>
  );
};
