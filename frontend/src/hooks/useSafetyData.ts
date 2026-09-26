import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  SystemStatusResponse,
  WorkerStateModel,
  EventModel,
  StatisticsResponse,
} from '../types/safety';
import { safetyApi } from '../services/api';

export interface SafetyDataState {
  status: SystemStatusResponse | null;
  workers: WorkerStateModel[];
  events: EventModel[];
  statistics: StatisticsResponse | null;
  isOnline: boolean;
  isLoading: boolean;
  lastUpdated: Date | null;
  error: string | null;
}

export function useSafetyData(pollIntervalMs = 2500) {
  const [data, setData] = useState<SafetyDataState>({
    status: null,
    workers: [],
    events: [],
    statistics: null,
    isOnline: false,
    isLoading: true,
    lastUpdated: null,
    error: null,
  });

  const isMountedRef = useRef(true);

  const fetchAllData = useCallback(async () => {
    try {
      const [statusRes, workersRes, eventsRes, statsRes] = await Promise.all([
        safetyApi.getStatus(),
        safetyApi.getWorkers(),
        safetyApi.getEvents(30, 0),
        safetyApi.getStatistics(),
      ]);

      if (!isMountedRef.current) return;

      setData({
        status: statusRes,
        workers: workersRes,
        events: eventsRes,
        statistics: statsRes,
        isOnline: true,
        isLoading: false,
        lastUpdated: new Date(),
        error: null,
      });
    } catch (err: any) {
      if (!isMountedRef.current) return;

      setData((prev) => ({
        ...prev,
        isOnline: false,
        isLoading: false,
        error: err?.message || 'Backend connection offline',
      }));
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;

    fetchAllData();
    const intervalId = setInterval(fetchAllData, pollIntervalMs);

    return () => {
      isMountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [fetchAllData, pollIntervalMs]);

  return {
    ...data,
    refreshNow: fetchAllData,
  };
}
