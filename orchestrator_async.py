#!/usr/bin/env python3
"""
⚡ ASYNC ORCHESTRATOR WITH CONCURRENT EXECUTION
Part of Make It Heavy - Async Performance Edition

This is the high-performance async orchestrator that coordinates multiple
specialized agents running in parallel with advanced features:
- Concurrent agent execution with rate limiting
- AI-powered question generation for specialized perspectives
- Intelligent result synthesis
- Real-time progress monitoring and metrics

Usage: Import AsyncTaskOrchestrator class
"""

import json
import yaml
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from agent_async import AsyncOpenRouterAgent
from utils import ColorPrint


@dataclass
class AgentResult:
    """Data class for agent execution results"""

    agent_id: int
    status: str
    response: str
    execution_time: float
    error: Optional[str] = None


class AsyncTaskOrchestrator:
    """High-performance async orchestrator with advanced features"""

    def __init__(self, config_path="config.yaml", silent=False):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.num_agents = self.config["orchestrator"]["parallel_agents"]
        self.task_timeout = self.config["orchestrator"]["task_timeout"]
        self.aggregation_strategy = self.config["orchestrator"]["aggregation_strategy"]
        self.silent = silent

        # Performance tracking
        self.agent_progress = {}
        self.agent_results = {}
        self.metrics = {
            "total_agents": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "total_execution_time": 0,
            "question_generation_time": 0,
            "synthesis_time": 0,
        }

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(self.num_agents)

    async def decompose_task(self, user_input: str, num_agents: int) -> List[str]:
        """Use AI to dynamically generate different questions based on user input"""
        start_time = time.time()

        try:
            # Create question generation agent with async context
            async with AsyncOpenRouterAgent(silent=True) as question_agent:
                # Get question generation prompt from config
                prompt_template = self.config["orchestrator"][
                    "question_generation_prompt"
                ]
                generation_prompt = prompt_template.format(
                    user_input=user_input, num_agents=num_agents
                )

                # Remove task completion tool to avoid issues
                question_agent.tools = [
                    tool
                    for tool in question_agent.tools
                    if tool.get("function", {}).get("name") != "mark_task_complete"
                ]
                question_agent.tool_mapping = {
                    name: func
                    for name, func in question_agent.tool_mapping.items()
                    if name != "mark_task_complete"
                }

                # Get AI-generated questions
                response = await question_agent.run(generation_prompt)

                # Parse JSON response
                questions = json.loads(response.strip())

                # Validate we got the right number of questions
                if len(questions) != num_agents:
                    raise ValueError(
                        f"Expected {num_agents} questions, got {len(questions)}"
                    )

                self.metrics["question_generation_time"] = time.time() - start_time
                return questions

        except (json.JSONDecodeError, ValueError) as e:
            ColorPrint.warning(
                f"Question generation failed: {e}, using fallback", self.silent
            )
            # Fallback: create simple variations if AI fails
            fallback_questions = [
                f"Research comprehensive information about: {user_input}",
                f"Analyze and provide insights about: {user_input}",
                f"Find alternative perspectives on: {user_input}",
                f"Verify and cross-check facts about: {user_input}",
            ][:num_agents]

            self.metrics["question_generation_time"] = time.time() - start_time
            return fallback_questions

    async def run_agent_with_semaphore(
        self, agent_id: int, subtask: str
    ) -> AgentResult:
        """Run a single agent with rate limiting"""
        async with self.rate_limiter:
            return await self.run_single_agent(agent_id, subtask)

    async def run_single_agent(self, agent_id: int, subtask: str) -> AgentResult:
        """Run a single agent with the given subtask"""
        start_time = time.time()

        try:
            self.agent_progress[agent_id] = "PROCESSING..."

            # Use async agent with context manager
            async with AsyncOpenRouterAgent(silent=True) as agent:
                response = await agent.run(subtask)
                execution_time = time.time() - start_time

                self.agent_progress[agent_id] = "COMPLETED"
                self.agent_results[agent_id] = response

                return AgentResult(
                    agent_id=agent_id,
                    status="success",
                    response=response,
                    execution_time=execution_time,
                )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Agent {agent_id + 1} timed out after {self.task_timeout}s"
            self.agent_progress[agent_id] = "TIMEOUT"

            return AgentResult(
                agent_id=agent_id,
                status="timeout",
                response=error_msg,
                execution_time=execution_time,
                error="timeout",
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent {agent_id + 1} failed: {str(e)}"
            self.agent_progress[agent_id] = "FAILED"

            return AgentResult(
                agent_id=agent_id,
                status="error",
                response=error_msg,
                execution_time=execution_time,
                error=str(e),
            )

    async def aggregate_results(self, agent_results: List[AgentResult]) -> str:
        """Combine results from all agents into a comprehensive final answer"""
        start_time = time.time()

        successful_results = [r for r in agent_results if r.status == "success"]

        if not successful_results:
            return "All agents failed to provide results. Please try again."

        # Extract responses for aggregation
        responses = [r.response for r in successful_results]

        if len(responses) == 1:
            self.metrics["synthesis_time"] = time.time() - start_time
            return responses[0]

        try:
            # Create synthesis agent to combine all responses
            async with AsyncOpenRouterAgent(silent=True) as synthesis_agent:
                # Build agent responses section
                agent_responses_text = ""
                for i, response in enumerate(responses, 1):
                    agent_responses_text += (
                        f"=== AGENT {i} RESPONSE ===\n{response}\n\n"
                    )

                # Get synthesis prompt from config and format it
                synthesis_prompt_template = self.config["orchestrator"][
                    "synthesis_prompt"
                ]
                synthesis_prompt = synthesis_prompt_template.format(
                    num_responses=len(responses), agent_responses=agent_responses_text
                )

                # Remove all tools from synthesis agent to force direct response
                synthesis_agent.tools = []
                synthesis_agent.tool_mapping = {}

                # Get the synthesized response
                final_answer = await synthesis_agent.run(synthesis_prompt)
                self.metrics["synthesis_time"] = time.time() - start_time
                return final_answer

        except Exception as e:
            ColorPrint.error(f"Synthesis failed: {str(e)}", self.silent)
            # Fallback: concatenate responses
            combined = []
            for i, response in enumerate(responses, 1):
                combined.append(f"=== Agent {i} Response ===")
                combined.append(response)
                combined.append("")

            self.metrics["synthesis_time"] = time.time() - start_time
            return "\n".join(combined)

    def get_progress_status(self) -> Dict[int, str]:
        """Get current progress status for all agents"""
        return self.agent_progress.copy()

    async def orchestrate(self, user_input: str) -> str:
        """Main orchestration method with full async support"""
        total_start_time = time.time()

        # Reset progress tracking
        self.agent_progress = {}
        self.agent_results = {}
        self.metrics["total_agents"] = self.num_agents

        try:
            # Decompose task into subtasks
            subtasks = await self.decompose_task(user_input, self.num_agents)

            # Initialize progress tracking
            for i in range(self.num_agents):
                self.agent_progress[i] = "QUEUED"

            # Create agent tasks with timeout
            agent_tasks = [
                asyncio.wait_for(
                    self.run_agent_with_semaphore(i, subtasks[i]),
                    timeout=self.task_timeout,
                )
                for i in range(self.num_agents)
            ]

            # Execute all agents concurrently
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    processed_results.append(
                        AgentResult(
                            agent_id=i,
                            status="error",
                            response=f"Agent {i + 1} failed: {str(result)}",
                            execution_time=self.task_timeout,
                            error=str(result),
                        )
                    )
                else:
                    processed_results.append(result)

            # Update metrics
            self.metrics["successful_agents"] = len(
                [r for r in processed_results if r.status == "success"]
            )
            self.metrics["failed_agents"] = len(
                [r for r in processed_results if r.status != "success"]
            )

            # Sort results by agent_id for consistent output
            processed_results.sort(key=lambda x: x.agent_id)

            # Aggregate results
            final_result = await self.aggregate_results(processed_results)

            self.metrics["total_execution_time"] = time.time() - total_start_time

            ColorPrint.metrics(
                f"Orchestration completed in {self.metrics['total_execution_time']:.2f}s",
                self.silent,
            )
            ColorPrint.metrics(
                f"Success rate: {self.metrics['successful_agents']}/{self.metrics['total_agents']}",
                self.silent,
            )

            return final_result

        except Exception as e:
            self.metrics["total_execution_time"] = time.time() - total_start_time
            ColorPrint.error(f"Orchestration failed: {str(e)}", self.silent)
            return f"Orchestration failed: {str(e)}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return self.metrics.copy()
