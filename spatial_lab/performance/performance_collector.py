"""
Performance Measurement Collection System for Spatial AI Research Lab

This module implements comprehensive performance data collection with real-time
monitoring, statistical tracking, and automated analysis capabilities.

Features:
- Real-time performance monitoring
- Multi-dimensional metric collection
- Statistical trend analysis
- Automated anomaly detection
- Performance benchmarking
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import threading
from queue import Queue
import sqlite3
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a single performance measurement."""
    metric_name: str
    value: float
    timestamp: datetime
    source: str  # e.g., 'coordinator', 'environment', 'agent_001'
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class PerformanceSnapshot:
    """Comprehensive performance snapshot at a point in time."""
    timestamp: datetime
    system_metrics: Dict[str, float]
    coordination_metrics: Dict[str, float] 
    environment_metrics: Dict[str, float]
    agent_metrics: Dict[str, Dict[str, float]]  # agent_id -> metrics
    derived_metrics: Dict[str, float]
    

@dataclass 
class PerformanceTrend:
    """Statistical trend analysis of performance metrics."""
    metric_name: str
    time_window: timedelta
    mean_value: float
    std_deviation: float
    trend_slope: float  # Linear trend slope
    trend_significance: float  # p-value of trend
    anomaly_count: int
    confidence_interval: Tuple[float, float]
    

class PerformanceCollector:
    """
    Comprehensive performance measurement collection system.
    
    Provides real-time monitoring, statistical analysis, and automated
    reporting of system performance metrics with scientific rigor.
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        storage_backend: str = "sqlite",
        storage_path: str = "performance_data.db",
        max_memory_samples: int = 10000,
        enable_real_time_analysis: bool = True
    ):
        self.collection_interval = collection_interval
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path)
        self.max_memory_samples = max_memory_samples
        self.enable_real_time_analysis = enable_real_time_analysis
        
        # Data storage
        self.metrics_queue = Queue()
        self.memory_buffer = deque(maxlen=max_memory_samples)
        self.performance_snapshots: List[PerformanceSnapshot] = []
        
        # Real-time tracking
        self.metric_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.registered_sources: Dict[str, Any] = {}
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistical analysis
        self.trend_analyzers: Dict[str, PerformanceTrend] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self.baseline_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Collection control
        self._collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Database setup
        if storage_backend == "sqlite":
            self._setup_sqlite_storage()
            
        logger.info(f"PerformanceCollector initialized with {collection_interval}s interval")
        
    def _setup_sqlite_storage(self):
        """Initialize SQLite database for performance data storage."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON performance_metrics(metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON performance_metrics(source)')
        
        # Create snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                snapshot_data TEXT NOT NULL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp ON performance_snapshots(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("SQLite storage initialized")
        
    def register_source(self, source_name: str, source_object: Any):
        """Register a performance data source."""
        self.registered_sources[source_name] = source_object
        logger.info(f"Registered performance source: {source_name}")
        
    def add_metric_callback(self, metric_name: str, callback: Callable[[PerformanceMetric], None]):
        """Add callback function for specific metric updates."""
        self.metric_callbacks[metric_name].append(callback)
        logger.info(f"Added callback for metric: {metric_name}")
        
    def set_anomaly_threshold(self, metric_name: str, threshold_std_dev: float = 3.0):
        """Set anomaly detection threshold for a metric."""
        self.anomaly_thresholds[metric_name] = threshold_std_dev
        logger.info(f"Set anomaly threshold for {metric_name}: {threshold_std_dev} std dev")
        
    def record_metric(
        self,
        metric_name: str,
        value: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a single performance metric."""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        # Add to queue for processing
        self.metrics_queue.put(metric)
        
        # Add to real-time stream
        self.metric_streams[metric_name].append((metric.timestamp, value))
        
        # Trigger callbacks
        for callback in self.metric_callbacks[metric_name]:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback for {metric_name}: {e}")
                
    def record_batch_metrics(self, metrics: List[Dict[str, Any]]):
        """Record multiple metrics at once."""
        for metric_data in metrics:
            self.record_metric(
                metric_data['name'],
                metric_data['value'],
                metric_data['source'],
                metric_data.get('metadata')
            )
            
    async def start_collection(self):
        """Start automated performance collection."""
        if self._collecting:
            logger.warning("Performance collection already running")
            return
            
        self._collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        if self.enable_real_time_analysis:
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            
        logger.info("Started performance collection")
        
    async def stop_collection(self):
        """Stop automated performance collection."""
        if not self._collecting:
            return
            
        self._collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
                
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
                
        # Process remaining metrics in queue
        await self._process_metrics_queue()
        
        logger.info("Stopped performance collection")
        
    async def _collection_loop(self):
        """Main collection loop."""
        while self._collecting:
            try:
                # Collect from registered sources
                await self._collect_from_sources()
                
                # Process metrics queue
                await self._process_metrics_queue()
                
                # Create performance snapshot
                snapshot = await self._create_performance_snapshot()
                if snapshot:
                    self.performance_snapshots.append(snapshot)
                    await self._store_snapshot(snapshot)
                    
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _analysis_loop(self):
        """Real-time analysis loop."""
        while self._collecting:
            try:
                # Update trend analysis
                await self._update_trend_analysis()
                
                # Check for anomalies
                await self._detect_anomalies()
                
                # Update baseline metrics
                await self._update_baselines()
                
                await asyncio.sleep(5.0)  # Analysis every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(5.0)
                
    async def _collect_from_sources(self):
        """Collect metrics from all registered sources."""
        for source_name, source_obj in self.registered_sources.items():
            try:
                # Try to get performance metrics from source
                if hasattr(source_obj, 'get_performance_metrics'):
                    metrics = source_obj.get_performance_metrics()
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                self.record_metric(
                                    f"{source_name}_{metric_name}",
                                    float(value),
                                    source_name
                                )
                                
                # Try coordination-specific metrics
                if hasattr(source_obj, 'agents') and hasattr(source_obj, 'active_tasks'):
                    # Coordination metrics
                    self.record_metric(
                        "active_agents",
                        len([a for a in source_obj.agents.values() if hasattr(a, 'available') and not a.available]),
                        source_name
                    )
                    self.record_metric(
                        "active_tasks",
                        len(source_obj.active_tasks),
                        source_name
                    )
                    
            except Exception as e:
                logger.error(f"Error collecting from source {source_name}: {e}")
                
    async def _process_metrics_queue(self):
        """Process all metrics in the queue."""
        processed_count = 0
        
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                
                # Add to memory buffer
                self.memory_buffer.append(metric)
                
                # Store to database
                await self._store_metric(metric)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
                break
                
        if processed_count > 0:
            logger.debug(f"Processed {processed_count} metrics")
            
    async def _store_metric(self, metric: PerformanceMetric):
        """Store metric to database."""
        if self.storage_backend == "sqlite":
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (metric_name, value, timestamp, source, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metric.metric_name,
                metric.value,
                metric.timestamp.isoformat(),
                metric.source,
                json.dumps(metric.metadata)
            ))
            
            conn.commit()
            conn.close()
            
    async def _create_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Create comprehensive performance snapshot."""
        if not self.memory_buffer:
            return None
            
        current_time = datetime.now()
        recent_window = current_time - timedelta(seconds=self.collection_interval * 2)
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.memory_buffer 
            if m.timestamp >= recent_window
        ]
        
        if not recent_metrics:
            return None
            
        # Group by category
        system_metrics = {}
        coordination_metrics = {}
        environment_metrics = {}
        agent_metrics = defaultdict(dict)
        
        for metric in recent_metrics:
            if metric.source.startswith('agent_'):
                agent_id = metric.source
                agent_metrics[agent_id][metric.metric_name] = metric.value
            elif 'coordination' in metric.source.lower():
                coordination_metrics[metric.metric_name] = metric.value
            elif 'environment' in metric.source.lower():
                environment_metrics[metric.metric_name] = metric.value
            else:
                system_metrics[metric.metric_name] = metric.value
                
        # Calculate derived metrics
        derived_metrics = {}
        if coordination_metrics.get('active_tasks', 0) > 0 and coordination_metrics.get('active_agents', 0) > 0:
            derived_metrics['tasks_per_agent'] = coordination_metrics['active_tasks'] / coordination_metrics['active_agents']
            
        if len(agent_metrics) > 0:
            # Average agent utilization
            utilizations = []
            for agent_data in agent_metrics.values():
                if 'utilization' in agent_data:
                    utilizations.append(agent_data['utilization'])
            if utilizations:
                derived_metrics['avg_agent_utilization'] = np.mean(utilizations)
                
        return PerformanceSnapshot(
            timestamp=current_time,
            system_metrics=system_metrics,
            coordination_metrics=coordination_metrics,
            environment_metrics=environment_metrics,
            agent_metrics=dict(agent_metrics),
            derived_metrics=derived_metrics
        )
        
    async def _store_snapshot(self, snapshot: PerformanceSnapshot):
        """Store performance snapshot."""
        if self.storage_backend == "sqlite":
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_snapshots (timestamp, snapshot_data)
                VALUES (?, ?)
            ''', (
                snapshot.timestamp.isoformat(),
                json.dumps(asdict(snapshot), default=str)
            ))
            
            conn.commit()
            conn.close()
            
    async def _update_trend_analysis(self):
        """Update statistical trend analysis for all metrics."""
        current_time = datetime.now()
        analysis_window = timedelta(minutes=10)  # Analyze last 10 minutes
        
        for metric_name, stream in self.metric_streams.items():
            if len(stream) < 10:  # Need minimum samples
                continue
                
            # Get recent data points
            recent_data = [
                (timestamp, value) for timestamp, value in stream
                if current_time - timestamp <= analysis_window
            ]
            
            if len(recent_data) < 5:
                continue
                
            values = [value for _, value in recent_data]
            timestamps = [(timestamp - recent_data[0][0]).total_seconds() for timestamp, _ in recent_data]
            
            # Calculate trend statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Linear trend analysis
            if len(values) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
            else:
                slope, p_value = 0.0, 1.0
                
            # Confidence interval
            ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
            ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))
            
            # Count anomalies
            threshold = self.anomaly_thresholds.get(metric_name, 3.0)
            anomaly_count = sum(1 for v in values if abs(v - mean_val) > threshold * std_val)
            
            # Update trend analysis
            self.trend_analyzers[metric_name] = PerformanceTrend(
                metric_name=metric_name,
                time_window=analysis_window,
                mean_value=mean_val,
                std_deviation=std_val,
                trend_slope=slope,
                trend_significance=p_value,
                anomaly_count=anomaly_count,
                confidence_interval=(ci_lower, ci_upper)
            )
            
    async def _detect_anomalies(self):
        """Detect performance anomalies."""
        for metric_name, trend in self.trend_analyzers.items():
            if trend.anomaly_count > 0:
                logger.warning(
                    f"Performance anomaly detected in {metric_name}: "
                    f"{trend.anomaly_count} anomalous values in {trend.time_window}"
                )
                
                # Trigger anomaly callbacks if registered
                if f"anomaly_{metric_name}" in self.metric_callbacks:
                    anomaly_metric = PerformanceMetric(
                        metric_name=f"anomaly_{metric_name}",
                        value=trend.anomaly_count,
                        timestamp=datetime.now(),
                        source="anomaly_detector",
                        metadata={
                            'mean': trend.mean_value,
                            'std': trend.std_deviation,
                            'threshold': self.anomaly_thresholds.get(metric_name, 3.0)
                        }
                    )
                    
                    for callback in self.metric_callbacks[f"anomaly_{metric_name}"]:
                        try:
                            callback(anomaly_metric)
                        except Exception as e:
                            logger.error(f"Error in anomaly callback: {e}")
                            
    async def _update_baselines(self):
        """Update baseline performance metrics."""
        for metric_name, stream in self.metric_streams.items():
            if len(stream) >= 100:  # Need sufficient data for baseline
                values = [value for _, value in stream]
                
                # Keep rolling baseline (last 1000 samples)
                self.baseline_metrics[metric_name] = values[-1000:]
                
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        current_metrics = {}
        
        for metric_name, stream in self.metric_streams.items():
            if stream:
                current_metrics[metric_name] = stream[-1][1]  # Latest value
                
        return current_metrics
        
    def get_metric_history(
        self,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> List[Tuple[datetime, float]]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metric_streams:
            return []
            
        stream = self.metric_streams[metric_name]
        
        if time_window is None:
            return list(stream)
            
        current_time = datetime.now()
        cutoff_time = current_time - time_window
        
        return [(timestamp, value) for timestamp, value in stream if timestamp >= cutoff_time]
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'collection_status': 'active' if self._collecting else 'stopped',
            'total_metrics_collected': len(self.memory_buffer),
            'active_metric_streams': len(self.metric_streams),
            'registered_sources': list(self.registered_sources.keys()),
            'current_metrics': self.get_current_metrics(),
            'trend_analysis': {},
            'anomaly_summary': {}
        }
        
        # Add trend analysis
        for metric_name, trend in self.trend_analyzers.items():
            summary['trend_analysis'][metric_name] = {
                'mean': trend.mean_value,
                'std_deviation': trend.std_deviation,
                'trend_slope': trend.trend_slope,
                'trend_significant': trend.trend_significance < 0.05,
                'confidence_interval': trend.confidence_interval
            }
            
        # Anomaly summary
        total_anomalies = sum(trend.anomaly_count for trend in self.trend_analyzers.values())
        summary['anomaly_summary'] = {
            'total_anomalies_detected': total_anomalies,
            'metrics_with_anomalies': [
                name for name, trend in self.trend_analyzers.items() 
                if trend.anomaly_count > 0
            ]
        }
        
        return summary
        
    async def export_data(
        self,
        output_file: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metrics_filter: Optional[List[str]] = None
    ) -> str:
        """Export collected performance data."""
        output_path = Path(output_file)
        
        # Query data from database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM performance_metrics"
        params = []
        
        if time_range:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params.extend([t.isoformat() for t in time_range])
            
        if metrics_filter:
            if time_range:
                query += " AND metric_name IN ({})".format(','.join(['?'] * len(metrics_filter)))
            else:
                query += " WHERE metric_name IN ({})".format(','.join(['?'] * len(metrics_filter)))
            params.extend(metrics_filter)
            
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Export to JSON
        export_data = []
        for row in results:
            export_data.append({
                'id': row[0],
                'metric_name': row[1],
                'value': row[2],
                'timestamp': row[3],
                'source': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            })
            
        with open(output_path, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(export_data),
                'time_range': [t.isoformat() for t in time_range] if time_range else None,
                'metrics_filter': metrics_filter,
                'data': export_data
            }, f, indent=2)
            
        conn.close()
        
        logger.info(f"Exported {len(export_data)} records to {output_path}")
        return str(output_path) 