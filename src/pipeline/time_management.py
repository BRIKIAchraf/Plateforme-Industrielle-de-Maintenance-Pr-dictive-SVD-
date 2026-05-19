from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
import yaml

class SparkTimeManager:
    """
    Handles event-time processing, Spark watermarking, multi-sensor synchronization
    and Spark structured streaming backpressure tuning.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.watermark_duration = self.config['spark']['watermark_duration']
        self.trigger_time = self.config['spark']['trigger_processing_time']
        self.backpressure_conf = self.config['spark']['backpressure']

    def configure_backpressure(self, spark_session) -> None:
        """
        Dynamically applies backpressure properties to the active SparkSession
        to prevent downstream saturation under high ingestion rates (10k+/sec).
        """
        sc = spark_session.sparkContext
        sc._conf.set("spark.streaming.backpressure.enabled", str(self.backpressure_conf['enabled']).lower())
        sc._conf.set("spark.streaming.kafka.maxRatePerPartition", str(self.backpressure_conf['max_rate']))
        sc._conf.set("spark.streaming.backpressure.initialRate", str(self.backpressure_conf['initial_rate']))

    def apply_event_time_processing(self, df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
        """
        Converts the raw timestamp string column into TimestampType and
        registers watermarking to allow Spark to discard extremely late data.
        """
        # Ensure timestamp column is parsed as timestamp type
        processed_df = df.withColumn(
            "event_time", 
            F.col(timestamp_col).cast(TimestampType())
        )
        
        # Apply watermarking
        return processed_df.withWatermark("event_time", self.watermark_duration)

    def synchronize_multi_sensors(
        self, 
        df: DataFrame, 
        window_duration: str = "5 seconds", 
        slide_duration: str = "5 seconds"
    ) -> DataFrame:
        """
        Groups multi-sensor records by an event-time window to align late and
        out-of-order data streams before running SVD analysis.
        """
        # Group by sliding or tumbling windows and pivot/aggregate sensor values
        synchronized_df = df \
            .groupBy(
                F.window(F.col("event_time"), window_duration, slide_duration),
                F.col("machine_id")
            ) \
            .agg(
                F.collect_list(
                    F.create_map(F.col("sensor_name"), F.col("value"))
                ).alias("sensors_data")
            )
            
        return synchronized_df
