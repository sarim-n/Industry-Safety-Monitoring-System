import type {
  HealthResponse,
  SystemStatusResponse,
  WorkerStateModel,
  EventModel,
  StatisticsResponse,
} from '../types/safety';

const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
        ...options?.headers,
      },
    });
    clearTimeout(id);

    if (!response.ok) {
      throw new Error(`API error HTTP ${response.status} on ${endpoint}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

export const safetyApi = {
  getHealth: () => fetchJson<HealthResponse>('/api/health'),
  getStatus: () => fetchJson<SystemStatusResponse>('/api/status'),
  getWorkers: () => fetchJson<WorkerStateModel[]>('/api/workers'),
  getEvents: (limit = 20, offset = 0) =>
    fetchJson<EventModel[]>(`/api/events?limit=${limit}&offset=${offset}`),
  getStatistics: () => fetchJson<StatisticsResponse>('/api/statistics'),
  getVideoStreamUrl: () => `${API_BASE_URL}/api/video/stream`,
  getEvidenceUrl: (evidencePath: string) => {
    if (!evidencePath) return '';
    const filename = evidencePath.split('/').pop()?.split('\\').pop() || evidencePath;
    return `${API_BASE_URL}/api/evidence/${encodeURIComponent(filename)}`;
  },
};
