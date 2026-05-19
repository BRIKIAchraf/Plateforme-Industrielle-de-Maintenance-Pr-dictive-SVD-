import numpy as np
import collections
from typing import Dict, List, Tuple, Optional
import threading

class RealTimeStreamBuffer:
    """
    Thread-safe buffer that aggregates raw asynchronous IoT streams into structured
    sliding windows ready for mathematical matrix computations (SVD).
    
    Accommodates data alignment, missing timestamps, and dynamic sliding windows.
    """
    def __init__(self, window_size_seconds: float = 5.0, max_time_steps: int = 100):
        self.window_size_seconds = window_size_seconds
        self.max_time_steps = max_time_steps
        
        # Buffer mapping: sensor_id -> list of (timestamp, vibration, temperature)
        self.buffers: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_time_steps)
        )
        
        self.lock = threading.Lock()

    def append(self, sensor_id: int, timestamp: float, vibration: float, temperature: float) -> None:
        """
        Appends a single sensor measurement to the queue.
        """
        with self.lock:
            # We filter out extremely late data outside window size if needed,
            # but for simplicity we rely on deque's maxlen to manage memory.
            self.buffers[sensor_id].append((timestamp, vibration, temperature))

    def get_feature_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """
        Aligns the buffered data to construct the Feature Matrix X of shape (n_sensors, n_time_steps)
        along with the list of sensor IDs.
        
        It handles out-of-sync and missing timestamps across sensors using linear interpolation
        or forward/backward fill to guarantee numerical completeness.
        """
        with self.lock:
            sensor_ids = sorted(list(self.buffers.keys()))
            if not sensor_ids:
                return np.empty((0, 0)), []
                
            # Determine the time grid. We sample the union of timestamps or build an even time grid.
            # To be numerically robust, we establish a common time grid based on the latest timestamps seen.
            all_timestamps = []
            for s_id in sensor_ids:
                all_timestamps.extend([item[0] for item in self.buffers[s_id]])
                
            if not all_timestamps:
                return np.empty((0, 0)), []
                
            min_t, max_t = min(all_timestamps), max(all_timestamps)
            
            # Form an evenly spaced grid of self.max_time_steps points
            time_grid = np.linspace(min_t, max_t, self.max_time_steps)
            
            # Construct matrix X of shape (n_sensors, n_time_steps) representing vibrations
            X = np.zeros((len(sensor_ids), self.max_time_steps))
            
            for i, s_id in enumerate(sensor_ids):
                records = list(self.buffers[s_id])
                times = np.array([r[0] for r in records])
                vibs = np.array([r[1] for r in records])
                
                # Filter out NaNs if any are present in the raw data
                nan_mask = np.isnan(vibs)
                if nan_mask.any():
                    # If all are NaN, default to zero
                    if nan_mask.all():
                        vibs = np.zeros_like(vibs)
                    else:
                        # Simple linear interpolation for NaNs in-place
                        vibs[nan_mask] = np.interp(
                            np.where(nan_mask)[0], 
                            np.where(~nan_mask)[0], 
                            vibs[~nan_mask]
                        )
                
                if len(times) < 2:
                    # Not enough historical points to interpolate, fill with last vibration or zero
                    val = vibs[0] if len(vibs) > 0 else 0.0
                    X[i, :] = val
                else:
                    # Interpolate sensor vibrations onto the common time grid
                    X[i, :] = np.interp(time_grid, times, vibs, left=vibs[0], right=vibs[-1])
                    
            return X, sensor_ids

    def clear(self) -> None:
        """
        Flushes the buffer state.
        """
        with self.lock:
            self.buffers.clear()
