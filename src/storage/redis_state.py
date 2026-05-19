import json
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("RedisState")

class RedisStateManager:
    """
    Manages real-time state storage, machine health scores, dynamic thresholds, 
    and alert logging in Redis. Includes simulation fallback if Redis is offline.
    """
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379, 
        password: str = "", 
        db: int = 0
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client = None
        self.connected = False
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    decode_responses=True,
                    socket_timeout=2.0
                )
                self.client.ping()
                self.connected = True
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Redis connection failed ({e}). Operating in local cache simulation mode.")
        else:
            logger.warning("Redis package is not installed. Operating in local cache simulation mode.")

        # Local cache fallback
        self._local_state: Dict[str, Any] = {}

    def save_metrics(
        self, 
        step: int, 
        anomaly_score: float, 
        threshold: float, 
        is_alarm: bool, 
        condition_number: float, 
        singular_values: np.ndarray,
        rul: float
    ) -> None:
        """
        Stores the current anomaly, SVD spectrum, and telemetry state in Redis.
        """
        state_data = {
            "timestamp": time.time(),
            "step": int(step),
            "anomaly_score": float(anomaly_score),
            "threshold": float(threshold),
            "is_alarm": int(is_alarm),
            "condition_number": float(condition_number),
            "singular_values": json.dumps(singular_values.tolist() if isinstance(singular_values, np.ndarray) else list(singular_values)),
            "estimated_rul": float(rul) if rul != float('inf') else 9999.0
        }
        
        if self.connected:
            try:
                # Store the latest step state
                self.client.hset(f"machine:state:latest", mapping=state_data)
                
                # If an alarm is triggered, store a persistent alarm log
                if is_alarm:
                    self.client.hset(f"alarm:{step}", mapping=state_data)
                    self.client.expire(f"alarm:{step}", 3600)  # Expire alarm logs after 1 hour
                    
                # Push anomaly score to a historical list for Grafana charts
                self.client.lpush("machine:history:scores", float(anomaly_score))
                self.client.ltrim("machine:history:scores", 0, 100)  # Keep last 100 scores
            except Exception as e:
                logger.error(f"Failed to save metrics to Redis: {e}")
        else:
            # Store in local fallback dictionary
            self._local_state["latest"] = state_data
            if is_alarm:
                self._local_state[f"alarm:{step}"] = state_data

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last written machine state.
        """
        if self.connected:
            try:
                data = self.client.hgetall("machine:state:latest")
                if data:
                    # Deserialize lists and numeric values
                    data["step"] = int(data["step"])
                    data["anomaly_score"] = float(data["anomaly_score"])
                    data["threshold"] = float(data["threshold"])
                    data["is_alarm"] = bool(int(data["is_alarm"]))
                    data["condition_number"] = float(data["condition_number"])
                    data["singular_values"] = json.loads(data["singular_values"])
                    data["estimated_rul"] = float(data["estimated_rul"])
                    return data
            except Exception as e:
                logger.error(f"Failed to fetch state from Redis: {e}")
                
        return self._local_state.get("latest")

    def close(self) -> None:
        """
        Closes the Redis connection.
        """
        if self.client:
            self.client.close()
