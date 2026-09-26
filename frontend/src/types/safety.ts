export type SafetyStatus = 
  | 'SAFE'
  | 'NO_HELMET'
  | 'NO_MASK'
  | 'NO_HELMET_AND_MASK'
  | 'UNCERTAIN';

export interface HealthResponse {
  status: string;
}

export interface WorkerStateModel {
  worker_id: number;
  status: SafetyStatus;
}

export interface EventModel {
  timestamp: string;
  worker_id: number;
  violation_type: SafetyStatus | string;
  evidence_path: string;
}

export interface SystemStatusResponse {
  system_running: boolean;
  active_workers: number;
  confirmed_violations: number;
  last_event: EventModel | null;
}

export interface StatisticsResponse {
  total_events: number;
  no_helmet_count: number;
  no_mask_count: number;
  no_helmet_and_mask_count: number;
  unique_workers: number;
  latest_event_timestamp: string | null;
}
