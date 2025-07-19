#!/usr/bin/env python3
"""
⚡ ASYNC MULTI-AGENT ORCHESTRATOR CLI
Part of Make It Heavy - Async Performance Edition

This is the high-performance async version of the multi-agent orchestrator
that runs 4 specialized agents in parallel with real-time progress monitoring,
connection pooling, and beautiful colorful output.

Usage: python make_it_heavy_async.py
"""

import asyncio
import time
import sys
from orchestrator_async import AsyncTaskOrchestrator
from utils import ColorPrint


class AsyncOrchestratorCLI:
    """High-performance async orchestrator CLI with real-time monitoring"""

    def __init__(self):
        self.orchestrator = AsyncTaskOrchestrator()
        self.start_time = None
        self.running = False
        self.progress_task = None

        # Extract model name for display
        model_full = self.orchestrator.config["openrouter"]["model"]
        if "/" in model_full:
            model_name = model_full.split("/")[-1]
        else:
            model_name = model_full

        # Clean up model name for display
        model_parts = model_name.split("-")
        clean_name = "-".join(model_parts[:3]) if len(model_parts) >= 3 else model_name
        self.model_display = clean_name.upper() + " HEAVY ASYNC"

    def clear_screen(self):
        """Clear the screen efficiently"""
        import os

        os.system("cls" if os.name == "nt" else "clear")

    def format_time(self, seconds):
        """Format seconds into readable time string"""
        if seconds < 60:
            return f"{int(seconds)}S"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}M{secs}S"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}H{minutes}M"

    def create_progress_bar(self, status, progress_percent=0):
        """Create enhanced progress visualization"""
        # ANSI color codes
        ORANGE = "\033[38;5;208m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"

        bar_length = 50

        if status == "QUEUED":
            return "○ " + "·" * bar_length
        elif status == "INITIALIZING...":
            filled = int(bar_length * 0.1)
            return (
                f"{BLUE}◐{RESET} "
                + f"{BLUE}▓{RESET}" * filled
                + "·" * (bar_length - filled)
            )
        elif status == "PROCESSING...":
            # Animated processing bar
            filled = int(bar_length * min(progress_percent / 100, 0.8))
            return (
                f"{ORANGE}●{RESET} "
                + f"{ORANGE}▓{RESET}" * filled
                + "·" * (bar_length - filled)
            )
        elif status == "COMPLETED":
            return f"{GREEN}●{RESET} " + f"{GREEN}▓{RESET}" * bar_length
        elif status in ["FAILED", "TIMEOUT"]:
            return f"{RED}✗{RESET} " + f"{RED}×{RESET}" * bar_length
        else:
            return f"{BLUE}◐{RESET} " + "·" * bar_length

    async def update_display(self):
        """Update the console display with current status"""
        if not self.running:
            return

        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0
        time_str = self.format_time(elapsed)

        # Get current progress
        progress = self.orchestrator.get_progress_status()

        # Clear screen
        self.clear_screen()

        # Header
        ColorPrint.header(self.model_display)
        status_text = "● RUNNING" if self.running else "● COMPLETED"
        ColorPrint.info(f"{status_text} • {time_str}")

        # Agent status lines with enhanced progress
        for i in range(self.orchestrator.num_agents):
            status = progress.get(i, "QUEUED")
            # Simulate progress percentage for processing agents
            progress_percent = min(elapsed * 10, 90) if status == "PROCESSING..." else 0
            progress_bar = self.create_progress_bar(status, progress_percent)
            ColorPrint.info(f"AGENT {i + 1:02d}  {progress_bar}", silent=False)

        # Performance metrics if available
        metrics = self.orchestrator.get_metrics()
        if metrics["total_agents"] > 0:
            ColorPrint.metrics(
                f"Success Rate: {metrics['successful_agents']}/{metrics['total_agents']}"
            )
            if metrics["question_generation_time"] > 0:
                ColorPrint.performance(
                    f"Question Gen: {metrics['question_generation_time']:.2f}s"
                )

        sys.stdout.flush()

    async def progress_monitor(self):
        """Monitor and update progress display"""
        while self.running:
            await self.update_display()
            await asyncio.sleep(0.5)  # Update every 500ms for smooth animation

    async def run_task(self, user_input):
        """Run orchestrator task with live progress display"""
        self.start_time = time.time()
        self.running = True

        # Start progress monitoring
        self.progress_task = asyncio.create_task(self.progress_monitor())

        try:
            # Run the orchestrator
            result = await self.orchestrator.orchestrate(user_input)

            # Stop progress monitoring
            self.running = False
            if self.progress_task:
                self.progress_task.cancel()
                try:
                    await self.progress_task
                except asyncio.CancelledError:
                    pass

            # Final display update
            await self.update_display()

            # Show comprehensive results
            metrics = self.orchestrator.get_metrics()
            ColorPrint.header("FINAL RESULTS")
            ColorPrint.success(result)
            ColorPrint.subheader("PERFORMANCE METRICS")
            ColorPrint.performance(
                f"Total Time: {metrics['total_execution_time']:.2f}s"
            )
            ColorPrint.metrics(
                f"Agents: {metrics['successful_agents']}/{metrics['total_agents']} successful"
            )
            ColorPrint.metrics(
                f"Question Generation: {metrics['question_generation_time']:.2f}s"
            )
            ColorPrint.metrics(f"Synthesis Time: {metrics['synthesis_time']:.2f}s")
            ColorPrint.performance(
                f"Success Rate: {(metrics['successful_agents'] / metrics['total_agents'] * 100):.1f}%"
            )

            return result

        except Exception as e:
            self.running = False
            if self.progress_task:
                self.progress_task.cancel()
                try:
                    await self.progress_task
                except asyncio.CancelledError:
                    pass

            await self.update_display()
            ColorPrint.error(f"Orchestration failed: {str(e)}")
            ColorPrint.error(f"Error during orchestration: {str(e)}")
            return None

    async def interactive_mode(self):
        """Run interactive CLI session"""
        ColorPrint.header("HIGH-PERFORMANCE MULTI-AGENT ORCHESTRATOR")
        ColorPrint.info(
            f"Configured for {self.orchestrator.num_agents} parallel async agents"
        )
        ColorPrint.info("Type 'quit', 'exit', or 'bye' to exit")
        ColorPrint.info("Type 'metrics' to see performance statistics")

        try:
            orchestrator_config = self.orchestrator.config["openrouter"]
            ColorPrint.info(f"Using model: {orchestrator_config['model']}")
            ColorPrint.success("Orchestrator initialized successfully!")
            ColorPrint.info(
                "Note: Make sure to set your OpenRouter API key in config.yaml"
            )
        except Exception as e:
            ColorPrint.error(f"Failed to initialize orchestrator: {e}")
            ColorPrint.info("Make sure you have:")
            ColorPrint.info("   1. Set your OpenRouter API key in config.yaml")
            ColorPrint.info(
                "   2. Installed all dependencies with: pip install -r requirements.txt"
            )
            return

        session_start = time.time()
        query_count = 0

        while True:
            try:
                # Get user input asynchronously
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n👤 User: ").strip()
                )

                if user_input.lower() in ["quit", "exit", "bye"]:
                    session_time = time.time() - session_start
                    ColorPrint.subheader("SESSION SUMMARY")
                    ColorPrint.metrics(f"Total time: {session_time:.2f}s")
                    ColorPrint.metrics(f"Queries processed: {query_count}")
                    ColorPrint.success("Goodbye!")
                    break

                if user_input.lower() == "metrics":
                    metrics = self.orchestrator.get_metrics()
                    session_time = time.time() - session_start
                    ColorPrint.subheader("PERFORMANCE METRICS")
                    ColorPrint.metrics(f"Session time: {session_time:.2f}s")
                    ColorPrint.metrics(f"Queries this session: {query_count}")
                    ColorPrint.metrics(f"Total agents run: {metrics['total_agents']}")
                    ColorPrint.success(
                        f"Successful agents: {metrics['successful_agents']}"
                    )
                    ColorPrint.error(f"Failed agents: {metrics['failed_agents']}")
                    ColorPrint.performance(
                        f"Total execution time: {metrics['total_execution_time']:.2f}s"
                    )
                    if query_count > 0:
                        ColorPrint.performance(
                            f"Avg query time: {metrics['total_execution_time'] / query_count:.2f}s"
                        )
                    continue

                if not user_input:
                    ColorPrint.warning("Please enter a question or command.")
                    continue

                ColorPrint.processing(
                    "Orchestrator: Starting high-performance multi-agent analysis..."
                )

                # Run task with live progress
                result = await self.run_task(user_input)
                query_count += 1

                if result is None:
                    ColorPrint.error("Task failed. Please try again.")

            except KeyboardInterrupt:
                ColorPrint.warning("Interrupted by user...")
                break
            except Exception as e:
                ColorPrint.error(f"Error in interactive mode: {e}")
                ColorPrint.error(f"Error: {e}")
                ColorPrint.info("Please try again or type 'quit' to exit.")


async def main():
    """Main entry point for the async orchestrator CLI"""
    cli = AsyncOrchestratorCLI()
    await cli.interactive_mode()


def run_async_orchestrator():
    """Wrapper to run async orchestrator"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ColorPrint.success("Goodbye!")
    except Exception as e:
        ColorPrint.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_async_orchestrator()
