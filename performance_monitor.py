#!/usr/bin/env python3
"""
📊 ADVANCED PERFORMANCE MONITORING
Part of Make It Heavy - Async Performance Edition

This module provides comprehensive performance tracking, metrics collection,
and benchmarking capabilities for the async agent system.

Usage:
- Import and use PerformanceMonitor class in your code
- Run directly: python performance_monitor.py
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""

    timestamp: str
    operation: str
    duration: float
    success: bool
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    cache_hit: bool = False
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Advanced performance monitoring and benchmarking system"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool,
        cache_hit: bool = False,
        error_message: str = None,
    ):
        """Record a performance metric"""
        metric = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration=duration,
            success=success,
            cache_hit=cache_hit,
            error_message=error_message,
        )
        self.metrics.append(metric)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics:
            return {"error": "No metrics recorded"}

        # Basic stats
        total_operations = len(self.metrics)
        successful_operations = len([m for m in self.metrics if m.success])
        failed_operations = total_operations - successful_operations

        # Duration stats
        durations = [m.duration for m in self.metrics]
        cache_hits = len([m for m in self.metrics if m.cache_hit])

        # Operation breakdown
        operation_stats = {}
        for metric in self.metrics:
            op = metric.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    "count": 0,
                    "total_time": 0,
                    "successes": 0,
                    "failures": 0,
                    "cache_hits": 0,
                }

            stats = operation_stats[op]
            stats["count"] += 1
            stats["total_time"] += metric.duration
            if metric.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            if metric.cache_hit:
                stats["cache_hits"] += 1

        # Calculate averages
        for op_stats in operation_stats.values():
            op_stats["avg_time"] = op_stats["total_time"] / op_stats["count"]
            op_stats["success_rate"] = op_stats["successes"] / op_stats["count"] * 100
            op_stats["cache_hit_rate"] = (
                op_stats["cache_hits"] / op_stats["count"] * 100
            )

        return {
            "session_id": self.session_id,
            "session_duration": time.time() - self.start_time,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": (successful_operations / total_operations) * 100,
            "cache_hit_rate": (cache_hits / total_operations) * 100,
            "duration_stats": {
                "min": min(durations),
                "max": max(durations),
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
            },
            "operation_breakdown": operation_stats,
        }

    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            filename = f"performance_metrics_{self.session_id}.json"

        export_data = {
            "summary": self.get_summary(),
            "detailed_metrics": [asdict(m) for m in self.metrics],
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        return filename

    def print_summary(self):
        """Print formatted performance summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"🆔 Session ID: {summary['session_id']}")
        print(f"⏱️  Session Duration: {summary['session_duration']:.2f}s")
        print(f"📈 Total Operations: {summary['total_operations']}")
        print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        print(f"🎯 Cache Hit Rate: {summary['cache_hit_rate']:.1f}%")

        print("\n⚡ Duration Statistics:")
        duration_stats = summary["duration_stats"]
        print(f"   Min: {duration_stats['min']:.3f}s")
        print(f"   Max: {duration_stats['max']:.3f}s")
        print(f"   Mean: {duration_stats['mean']:.3f}s")
        print(f"   Median: {duration_stats['median']:.3f}s")
        print(f"   Std Dev: {duration_stats['std_dev']:.3f}s")

        print("\n🔧 Operation Breakdown:")
        for op_name, op_stats in summary["operation_breakdown"].items():
            print(f"   {op_name}:")
            print(f"     Count: {op_stats['count']}")
            print(f"     Avg Time: {op_stats['avg_time']:.3f}s")
            print(f"     Success Rate: {op_stats['success_rate']:.1f}%")
            print(f"     Cache Hit Rate: {op_stats['cache_hit_rate']:.1f}%")

        print("=" * 60)


class PerformanceContext:
    """Context manager for automatic performance tracking"""

    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        self.start_time = None
        self.cache_hit = False

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        self.monitor.record_metric(
            operation=self.operation,
            duration=duration,
            success=success,
            cache_hit=self.cache_hit,
            error_message=error_message,
        )

    def mark_cache_hit(self):
        """Mark this operation as a cache hit"""
        self.cache_hit = True


# Global performance monitor instance
global_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return global_monitor


def track_performance(operation: str):
    """Decorator for automatic performance tracking"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with PerformanceContext(global_monitor, operation) as ctx:
                    result = await func(*args, **kwargs)
                    return result

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with PerformanceContext(global_monitor, operation) as ctx:
                    result = func(*args, **kwargs)
                    return result

            return sync_wrapper

    return decorator


async def benchmark_agent_performance(
    agent_class, test_queries: List[str], iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark agent performance with multiple test queries"""
    print("🚀 Starting performance benchmark...")
    print(f"📝 Test queries: {len(test_queries)}")
    print(f"🔄 Iterations per query: {iterations}")

    benchmark_monitor = PerformanceMonitor()
    results = []

    for i, query in enumerate(test_queries):
        print(f"\n📊 Testing query {i + 1}/{len(test_queries)}: {query[:50]}...")

        query_results = []
        for iteration in range(iterations):
            try:
                async with agent_class() as agent:
                    start_time = time.time()
                    response = await agent.run(query)
                    duration = time.time() - start_time

                    query_results.append(
                        {
                            "iteration": iteration + 1,
                            "duration": duration,
                            "success": True,
                            "response_length": len(response),
                            "metrics": agent.get_metrics()
                            if hasattr(agent, "get_metrics")
                            else {},
                        }
                    )

                    benchmark_monitor.record_metric(
                        operation=f"query_{i + 1}", duration=duration, success=True
                    )

            except Exception as e:
                query_results.append(
                    {
                        "iteration": iteration + 1,
                        "duration": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

                benchmark_monitor.record_metric(
                    operation=f"query_{i + 1}",
                    duration=0,
                    success=False,
                    error_message=str(e),
                )

        # Calculate query statistics
        successful_runs = [r for r in query_results if r["success"]]
        if successful_runs:
            durations = [r["duration"] for r in successful_runs]
            query_stats = {
                "query": query,
                "success_rate": len(successful_runs) / len(query_results) * 100,
                "avg_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
            }
        else:
            query_stats = {
                "query": query,
                "success_rate": 0,
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0,
                "std_dev": 0,
            }

        results.append(query_stats)
        print(f"   ✅ Success rate: {query_stats['success_rate']:.1f}%")
        print(f"   ⚡ Avg duration: {query_stats['avg_duration']:.2f}s")

    # Generate comprehensive benchmark report
    benchmark_summary = benchmark_monitor.get_summary()

    print("\n🎯 BENCHMARK COMPLETE")
    benchmark_monitor.print_summary()

    return {
        "benchmark_summary": benchmark_summary,
        "query_results": results,
        "export_file": benchmark_monitor.export_metrics(
            f"benchmark_{benchmark_monitor.session_id}.json"
        ),
    }


if __name__ == "__main__":
    # Example usage
    async def example_benchmark():
        from agent_async import AsyncOpenRouterAgent

        test_queries = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "What are the latest developments in AI?",
        ]

        results = await benchmark_agent_performance(AsyncOpenRouterAgent, test_queries)
        print(f"\n📄 Benchmark results exported to: {results['export_file']}")

    asyncio.run(example_benchmark())
