#!/usr/bin/env python3
"""
⚡ HIGH-PERFORMANCE ASYNC SINGLE AGENT CLI
Part of Make It Heavy - Async Performance Edition

This is the main entry point for running a single intelligent agent
with full async performance optimizations including connection pooling,
caching, and beautiful colorful output.

Usage: python main_async.py
"""

import asyncio
import time
from agent_async import AsyncOpenRouterAgent
from utils import ColorPrint


async def main():
    """High-performance async main entry point"""
    ColorPrint.header("HIGH-PERFORMANCE ASYNC OPENROUTER AGENT")
    ColorPrint.info("Type 'quit', 'exit', or 'bye' to exit")
    ColorPrint.info("Type 'metrics' to see performance statistics")

    try:
        # Use async context manager for proper resource management
        async with AsyncOpenRouterAgent() as agent:
            ColorPrint.success("Agent initialized successfully!")
            ColorPrint.info(f"Using model: {agent.config['openrouter']['model']}")
            ColorPrint.info(
                "Note: Make sure to set your OpenRouter API key in config.yaml"
            )

            session_start = time.time()
            query_count = 0

            while True:
                try:
                    # Get user input (this blocks, but that's expected for CLI)
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\n👤 User: ").strip()
                    )

                    if user_input.lower() in ["quit", "exit", "bye"]:
                        session_time = time.time() - session_start
                        ColorPrint.subheader("SESSION SUMMARY")
                        ColorPrint.metrics(f"Total time: {session_time:.2f}s")
                        ColorPrint.metrics(f"Queries processed: {query_count}")
                        ColorPrint.metrics(f"Agent metrics: {agent.get_metrics()}")
                        ColorPrint.success("Goodbye!")
                        break

                    if user_input.lower() == "metrics":
                        metrics = agent.get_metrics()
                        session_time = time.time() - session_start
                        ColorPrint.subheader("PERFORMANCE METRICS")
                        ColorPrint.metrics(f"Session time: {session_time:.2f}s")
                        ColorPrint.metrics(f"Queries this session: {query_count}")
                        ColorPrint.metrics(f"LLM calls: {metrics['llm_calls']}")
                        ColorPrint.metrics(f"Tool calls: {metrics['tool_calls']}")
                        ColorPrint.cache(f"Cache hits: {metrics['cache_hits']}")
                        ColorPrint.performance(
                            f"Total processing time: {metrics['total_time']:.2f}s"
                        )
                        if query_count > 0:
                            ColorPrint.performance(
                                f"Avg response time: {metrics['total_time'] / query_count:.2f}s"
                            )
                        continue

                    if not user_input:
                        ColorPrint.warning("Please enter a question or command.")
                        continue

                    ColorPrint.processing("Agent: Processing your request...")
                    query_start = time.time()

                    # Process the request asynchronously
                    response = await agent.run(user_input)

                    query_time = time.time() - query_start
                    query_count += 1

                    ColorPrint.success(f"Agent: {response}")
                    ColorPrint.performance(f"Response time: {query_time:.2f}s")

                except KeyboardInterrupt:
                    ColorPrint.warning("Interrupted by user...")
                    break
                except Exception as e:
                    ColorPrint.error(f"Error processing request: {e}")
                    ColorPrint.error(f"Error: {e}")
                    ColorPrint.info("Please try again or type 'quit' to exit.")

    except Exception as e:
        ColorPrint.error(f"Failed to initialize agent: {e}")
        ColorPrint.info("Make sure you have:")
        ColorPrint.info("   1. Set your OpenRouter API key in config.yaml")
        ColorPrint.info(
            "   2. Installed all dependencies with: pip install -r requirements.txt"
        )
        return


def run_async_main():
    """Wrapper to run async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ColorPrint.success("Goodbye!")
    except Exception as e:
        ColorPrint.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_async_main()
