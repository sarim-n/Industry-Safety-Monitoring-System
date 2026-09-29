// Types for the Video Analysis Upload feature

export interface VideoUploadResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface VideoStatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_frame: number;
  total_frames: number;
  error: string | null;
}

export interface VideoResultsResponse {
  job_id: string;
  status: string;
  error: string | null;
  frames_processed: number;
  processing_fps: number;
  workers_detected: number;
  confirmed_violations: number;
  no_helmet: number;
  no_mask: number;
  no_helmet_and_mask: number;
  evidence_count: number;
  codec_used: string;
  model: string;
}
