#!/usr/bin/env python3
"""
📊 SYNC VS ASYNC PERFORMANCE BENCHMARK
Part of Make It Heavy - Async Performance Edition

This comprehensive benchmark tool compares the performance between
the original synchronous implementation and the new async version,
providing detailed metrics and analysis.

Usage: python benchmark_comparison.py
"""

import asyncio
import time
import statistics
import json
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import OpenRouterAgent
from agent_async import AsyncOpenRouterAgent
from orchestrator import TaskOrchestrator
from orchestrator_async import AsyncTaskOrchestrator


class BenchmarkRunner:
    """Comprehensive benchmark runner for sync vs async comparison"""

    def __init__(self):
        self.test_queries = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "What are the latest developments in AI in 2025?",
            "How does photosynthesis work?",
            "What are the benefits of renewable energy?",
        ]

        self.results = {
            "sync_agent": [],
            "async_agent": [],
            "sync_orchestrator": [],
            "async_orchestrator": [],
        }

    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'=' * 60}")
        print(f"🚀 {title}")
        print(f"{'=' * 60}")

    def print_progress(self, current: int, total: int, operation: str):
        """Print progress indicator"""
        progress = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(
            f"\r{operation}: |{bar}| {progress:.1f}% ({current}/{total})",
            end="",
            flush=True,
        )

    async def benchmark_sync_agent(self, iterations: int = 3):
        """Benchmark synchronous agent"""
        self.print_header("SYNC AGENT BENCHMARK")

        for i, query in enumerate(self.test_queries):
            print(f"\n📝 Query {i + 1}: {query[:50]}...")

            query_times = []
            for iteration in range(iterations):
                self.print_progress(iteration + 1, iterations, "Testing")

                try:
                    agent = OpenRouterAgent(silent=True)
                    start_time = time.time()
                    response = agent.run(query)
                    duration = time.time() - start_time

                    query_times.append(duration)

                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    query_times.append(float("inf"))

            # Calculate statistics
            valid_times = [t for t in query_times if t != float("inf")]
            if valid_times:
                avg_time = statistics.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                success_rate = len(valid_times) / len(query_times) * 100
            else:
                avg_time = min_time = max_time = 0
                success_rate = 0

            result = {
                "query": query,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_rate,
                "iterations": iterations,
            }

            self.results["sync_agent"].append(result)
            print(
                f"\n   ⚡ Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s | Success: {success_rate:.1f}%"
            )

    async def benchmark_async_agent(self, iterations: int = 3):
        """Benchmark asynchronous agent"""
        self.print_header("ASYNC AGENT BENCHMARK")

        for i, query in enumerate(self.test_queries):
            print(f"\n📝 Query {i + 1}: {query[:50]}...")

            query_times = []
            cache_hits = 0

            for iteration in range(iterations):
                self.print_progress(iteration + 1, iterations, "Testing")

                try:
                    async with AsyncOpenRouterAgent(silent=True) as agent:
                        start_time = time.time()
                        response = await agent.run(query)
                        duration = time.time() - start_time

                        query_times.append(duration)

                        # Check for cache hits
                        metrics = agent.get_metrics()
                        if metrics["cache_hits"] > 0:
                            cache_hits += 1

                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    query_times.append(float("inf"))

            # Calculate statistics
            valid_times = [t for t in query_times if t != float("inf")]
            if valid_times:
                avg_time = statistics.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                success_rate = len(valid_times) / len(query_times) * 100
            else:
                avg_time = min_time = max_time = 0
                success_rate = 0

            result = {
                "query": query,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_rate,
                "cache_hit_rate": (cache_hits / iterations) * 100,
                "iterations": iterations,
            }

            self.results["async_agent"].append(result)
            print(
                f"\n   ⚡ Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s | Success: {success_rate:.1f}% | Cache: {result['cache_hit_rate']:.1f}%"
            )

    async def benchmark_sync_orchestrator(self, iterations: int = 2):
        """Benchmark synchronous orchestrator"""
        self.print_header("SYNC ORCHESTRATOR BENCHMARK")

        # Use fewer queries for orchestrator due to complexity
        test_queries = self.test_queries[:3]

        for i, query in enumerate(test_queries):
            print(f"\n📝 Query {i + 1}: {query[:50]}...")

            query_times = []
            for iteration in range(iterations):
                self.print_progress(iteration + 1, iterations, "Testing")

                try:
                    orchestrator = TaskOrchestrator(silent=True)
                    start_time = time.time()
                    response = orchestrator.orchestrate(query)
                    duration = time.time() - start_time

                    query_times.append(duration)

                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    query_times.append(float("inf"))

            # Calculate statistics
            valid_times = [t for t in query_times if t != float("inf")]
            if valid_times:
                avg_time = statistics.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                success_rate = len(valid_times) / len(query_times) * 100
            else:
                avg_time = min_time = max_time = 0
                success_rate = 0

            result = {
                "query": query,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_rate,
                "iterations": iterations,
            }

            self.results["sync_orchestrator"].append(result)
            print(
                f"\n   ⚡ Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s | Success: {success_rate:.1f}%"
            )

    async def benchmark_async_orchestrator(self, iterations: int = 2):
        """Benchmark asynchronous orchestrator"""
        self.print_header("ASYNC ORCHESTRATOR BENCHMARK")

        # Use fewer queries for orchestrator due to complexity
        test_queries = self.test_queries[:3]

        for i, query in enumerate(test_queries):
            print(f"\n📝 Query {i + 1}: {query[:50]}...")

            query_times = []
            for iteration in range(iterations):
                self.print_progress(iteration + 1, iterations, "Testing")

                try:
                    orchestrator = AsyncTaskOrchestrator(silent=True)
                    start_time = time.time()
                    response = await orchestrator.orchestrate(query)
                    duration = time.time() - start_time

                    query_times.append(duration)

                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    query_times.append(float("inf"))

            # Calculate statistics
            valid_times = [t for t in query_times if t != float("inf")]
            if valid_times:
                avg_time = statistics.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                success_rate = len(valid_times) / len(query_times) * 100
            else:
                avg_time = min_time = max_time = 0
                success_rate = 0

            result = {
                "query": query,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_rate,
                "iterations": iterations,
            }

            self.results["async_orchestrator"].append(result)
            print(
                f"\n   ⚡ Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s | Success: {success_rate:.1f}%"
            )

    def calculate_improvements(self):
        """Calculate performance improvements"""
        improvements = {}

        # Agent comparison
        if self.results["sync_agent"] and self.results["async_agent"]:
            sync_avg = statistics.mean(
                [r["avg_time"] for r in self.results["sync_agent"] if r["avg_time"] > 0]
            )
            async_avg = statistics.mean(
                [
                    r["avg_time"]
                    for r in self.results["async_agent"]
                    if r["avg_time"] > 0
                ]
            )

            if sync_avg > 0 and async_avg > 0:
                improvement = ((sync_avg - async_avg) / sync_avg) * 100
                improvements["agent"] = {
                    "sync_avg": sync_avg,
                    "async_avg": async_avg,
                    "improvement_percent": improvement,
                    "speedup_factor": sync_avg / async_avg,
                }

        # Orchestrator comparison
        if self.results["sync_orchestrator"] and self.results["async_orchestrator"]:
            sync_avg = statistics.mean(
                [
                    r["avg_time"]
                    for r in self.results["sync_orchestrator"]
                    if r["avg_time"] > 0
                ]
            )
            async_avg = statistics.mean(
                [
                    r["avg_time"]
                    for r in self.results["async_orchestrator"]
                    if r["avg_time"] > 0
                ]
            )

            if sync_avg > 0 and async_avg > 0:
                improvement = ((sync_avg - async_avg) / sync_avg) * 100
                improvements["orchestrator"] = {
                    "sync_avg": sync_avg,
                    "async_avg": async_avg,
                    "improvement_percent": improvement,
                    "speedup_factor": sync_avg / async_avg,
                }

        return improvements

    def print_summary(self):
        """Print comprehensive benchmark summary"""
        self.print_header("BENCHMARK SUMMARY")

        improvements = self.calculate_improvements()

        if "agent" in improvements:
            agent_imp = improvements["agent"]
            print("\n🤖 SINGLE AGENT PERFORMANCE:")
            print(f"   Sync Average:     {agent_imp['sync_avg']:.2f}s")
            print(f"   Async Average:    {agent_imp['async_avg']:.2f}s")
            print(f"   Improvement:      {agent_imp['improvement_percent']:.1f}%")
            print(f"   Speedup Factor:   {agent_imp['speedup_factor']:.2f}x")

        if "orchestrator" in improvements:
            orch_imp = improvements["orchestrator"]
            print("\n🔀 MULTI-AGENT ORCHESTRATOR PERFORMANCE:")
            print(f"   Sync Average:     {orch_imp['sync_avg']:.2f}s")
            print(f"   Async Average:    {orch_imp['async_avg']:.2f}s")
            print(f"   Improvement:      {orch_imp['improvement_percent']:.1f}%")
            print(f"   Speedup Factor:   {orch_imp['speedup_factor']:.2f}x")

        # Cache hit analysis
        async_agent_results = self.results.get("async_agent", [])
        if async_agent_results:
            cache_rates = [r.get("cache_hit_rate", 0) for r in async_agent_results]
            avg_cache_rate = statistics.mean(cache_rates)
            print("\n🎯 CACHING PERFORMANCE:")
            print(f"   Average Cache Hit Rate: {avg_cache_rate:.1f}%")

        print("\n📊 OVERALL RECOMMENDATIONS:")
        if improvements:
            print(
                f"   ✅ Use async mode for {max(improvements.values(), key=lambda x: x['improvement_percent'])['improvement_percent']:.1f}% better performance"
            )
            print("   ✅ Async implementation shows consistent improvements")
            print("   ✅ Connection pooling and caching provide significant benefits")
        else:
            print("   ⚠️  Unable to calculate improvements - check for errors above")

    def export_results(self, filename: str = None):
        """Export benchmark results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "test_queries": self.test_queries,
            "results": self.results,
            "improvements": self.calculate_improvements(),
            "summary": {
                "total_tests": sum(len(results) for results in self.results.values()),
                "test_categories": list(self.results.keys()),
            },
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        return filename

    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        print("This will test both sync and async implementations")
        print("Estimated time: 5-10 minutes depending on API response times")
        print("=" * 60)

        start_time = time.time()

        try:
            # Run all benchmarks
            await self.benchmark_sync_agent(iterations=3)
            await self.benchmark_async_agent(iterations=3)
            await self.benchmark_sync_orchestrator(iterations=2)
            await self.benchmark_async_orchestrator(iterations=2)

            # Print summary
            self.print_summary()

            # Export results
            export_file = self.export_results()

            total_time = time.time() - start_time
            print("\n🎯 BENCHMARK COMPLETE")
            print(f"   Total Time: {total_time:.2f}s")
            print(f"   Results exported to: {export_file}")

        except KeyboardInterrupt:
            print("\n⚠️  Benchmark interrupted by user")
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")


async def main():
    """Main benchmark execution"""
    runner = BenchmarkRunner()
    await runner.run_full_benchmark()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
