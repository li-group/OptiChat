"""
Runtime tracking module for OptiChat agents.

Tracks execution time for:
- Agent executions (root, expert, illustrator)
- Operations (LLM call + resulting tool call)

Each operation represents:
  LLM reasoning (deciding what to do, generating code/args) + Tool execution (running the tool)
This shows the TRUE COST of each tool call including the LLM time to decide to use it.
"""

import time
import json
from typing import Dict, Any, Optional, List
from loguru import logger
from datetime import datetime


class TimingTracker:
    """Centralized timing tracker for agents and their operations.

    An operation = LLM call (reasoning/code generation) + optional tool call(s)
    This pairs each LLM call with its resulting tool execution to show true costs.
    """

    def __init__(self):
        self.timings: Dict[str, Any] = {
            "query_start_time": None,
            "query_end_time": None,
            "total_time": 0,
            "agents": {}
        }
        self.active_timers: Dict[str, float] = {}
        self.current_operations: Dict[str, Dict[str, Any]] = {}  # agent_name -> current operation being built

    def start_query(self):
        """Start timing a new user query."""
        self.timings["query_start_time"] = time.time()
        self.timings["agents"] = {}
        self.active_timers = {}

    def end_query(self):
        """End timing for the current query and finalize all operations."""
        # Finalize any remaining operations
        for agent_name in list(self.current_operations.keys()):
            self._finalize_current_operation(agent_name)

        if self.timings["query_start_time"]:
            self.timings["query_end_time"] = time.time()
            self.timings["total_time"] = (
                self.timings["query_end_time"] - self.timings["query_start_time"]
            )

    def start_timer(self, key: str):
        """Start a timer with the given key."""
        self.active_timers[key] = time.time()

    def end_timer(self, key: str) -> float:
        """End a timer and return elapsed time."""
        if key in self.active_timers:
            elapsed = time.time() - self.active_timers[key]
            del self.active_timers[key]
            return elapsed
        return 0.0

    def record_agent_start(self, agent_name: str):
        """Record the start of an agent execution."""
        if agent_name not in self.timings["agents"]:
            self.timings["agents"][agent_name] = {
                "total_time": 0,
                "operations": [],
                "delegations": [],
                "start_time": time.time(),
                "end_time": None
            }
        else:
            self.timings["agents"][agent_name]["start_time"] = time.time()

        # Initialize current operation tracker for this agent
        self.current_operations[agent_name] = None

    def record_agent_end(self, agent_name: str):
        """Record the end of an agent execution."""
        if agent_name in self.timings["agents"]:
            agent_data = self.timings["agents"][agent_name]
            agent_data["end_time"] = time.time()
            if agent_data["start_time"]:
                agent_data["total_time"] = agent_data["end_time"] - agent_data["start_time"]
                logger.info(
                    f"⏱️  [{agent_name}] Total runtime: {agent_data['total_time']:.3f}s "
                    f"({agent_data['total_time']/60:.2f}min)"
                )

    def _finalize_current_operation(self, agent_name: str):
        """Finalize the current operation for an agent and add to operations list."""
        if agent_name not in self.current_operations or self.current_operations[agent_name] is None:
            return

        operation = self.current_operations[agent_name]

        # Calculate total duration
        tool_time = sum(t["duration"] for t in operation.get("tool_calls", []))
        operation["total_duration"] = operation["llm_duration"] + tool_time

        # Add to operations list
        self.timings["agents"][agent_name]["operations"].append(operation)

        # Clear current operation
        self.current_operations[agent_name] = None

    def record_llm_call(self, agent_name: str, duration: float, call_number: int):
        """Record an LLM call, starting a new operation.

        This finalizes the previous operation (if any) and starts a new one.
        The operation will be completed when tool calls are added or the next LLM call starts.
        """
        if agent_name not in self.timings["agents"]:
            self.record_agent_start(agent_name)

        # Finalize previous operation
        self._finalize_current_operation(agent_name)

        # Start new operation
        operation_number = len(self.timings["agents"][agent_name]["operations"]) + 1
        self.current_operations[agent_name] = {
            "operation_number": operation_number,
            "llm_call_number": call_number,
            "llm_duration": duration,
            "tool_calls": [],
            "timestamp": time.time()
        }

        logger.info(
            f"⏱️  [{agent_name}] Operation #{operation_number}: LLM call #{call_number}: {duration:.3f}s"
        )

    def record_tool_call(self, agent_name: str, tool_name: str, duration: float):
        """Record a tool call, adding it to the current operation.

        This adds the tool call to the current operation started by the most recent LLM call.
        Multiple tools can be added to one operation (parallel tool calling).
        """
        if agent_name not in self.timings["agents"]:
            self.record_agent_start(agent_name)

        # Add tool to current operation
        if agent_name in self.current_operations and self.current_operations[agent_name] is not None:
            self.current_operations[agent_name]["tool_calls"].append({
                "tool_name": tool_name,
                "duration": duration,
                "timestamp": time.time()
            })

            logger.info(
                f"⏱️  [{agent_name}] Tool '{tool_name}': {duration:.3f}s (added to operation #{self.current_operations[agent_name]['operation_number']})"
            )
        else:
            # This shouldn't happen normally, but handle gracefully
            logger.warning(
                f"⏱️  [{agent_name}] Tool '{tool_name}': {duration:.3f}s (no current operation - tool call orphaned)"
            )

    def record_delegation(self, from_agent: str, to_agent: str, duration: float):
        """Record delegation from one agent to another."""
        if from_agent not in self.timings["agents"]:
            self.record_agent_start(from_agent)

        self.timings["agents"][from_agent]["delegations"].append({
            "to_agent": to_agent,
            "duration": duration,
            "timestamp": time.time()
        })

        logger.info(
            f"⏱️  [{from_agent}] Delegation to '{to_agent}': {duration:.3f}s"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timings."""
        summary = {
            "total_query_time": self.timings.get("total_time", 0),
            "agents": {}
        }

        for agent_name, agent_data in self.timings["agents"].items():
            operations = agent_data.get("operations", [])

            # Calculate totals from operations
            total_llm_time = sum(op["llm_duration"] for op in operations)
            total_tool_time = sum(
                sum(t["duration"] for t in op.get("tool_calls", []))
                for op in operations
            )
            total_tool_calls = sum(len(op.get("tool_calls", [])) for op in operations)
            total_delegation_time = sum(d["duration"] for d in agent_data.get("delegations", []))

            summary["agents"][agent_name] = {
                "total_time": agent_data["total_time"],
                "operations_count": len(operations),
                "llm_time": total_llm_time,
                "llm_calls_count": len(operations),  # Each operation has one LLM call
                "tool_time": total_tool_time,
                "tool_calls_count": total_tool_calls,
                "delegation_time": total_delegation_time,
                "delegations_count": len(agent_data.get("delegations", []))
            }

        return summary

    def print_summary(self):
        """Print a formatted summary of timings showing operations."""
        summary = self.get_summary()

        logger.info("\n" + "="*60)
        logger.info("⏱️  RUNTIME SUMMARY (Operation-Based)")
        logger.info("="*60)
        logger.info(f"Total Query Time: {summary['total_query_time']:.3f}s ({summary['total_query_time']/60:.2f}min)")
        logger.info("")

        for agent_name, agent_summary in summary["agents"].items():
            logger.info(f"📊 {agent_name.upper()}")
            logger.info(f"  Total Time: {agent_summary['total_time']:.3f}s")
            logger.info(f"  Operations: {agent_summary['operations_count']} (LLM call + tool execution)")
            logger.info(f"    ├─ LLM reasoning: {agent_summary['llm_time']:.3f}s ({agent_summary['llm_time']/agent_summary['total_time']*100:.1f}%)")
            logger.info(f"    └─ Tool execution: {agent_summary['tool_time']:.3f}s ({agent_summary['tool_time']/agent_summary['total_time']*100:.1f}%)")
            logger.info(f"  Total Tool Calls: {agent_summary['tool_calls_count']}")
            if agent_summary['delegations_count'] > 0:
                logger.info(f"  Delegations: {agent_summary['delegations_count']} delegations, {agent_summary['delegation_time']:.3f}s total")
            logger.info("")

        logger.info("="*60)

    def save_to_file(self, filepath: str):
        """Save detailed timing data to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": timestamp,
            "summary": self.get_summary(),
            "detailed_timings": self.timings
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"✅ Timing data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save timing data: {e}")

    def get_detailed_timings(self) -> Dict[str, Any]:
        """Get the complete detailed timing data."""
        return self.timings


# Global tracker instance
_global_tracker: Optional[TimingTracker] = None


def get_tracker() -> TimingTracker:
    """Get or create the global timing tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TimingTracker()
    return _global_tracker


def reset_tracker():
    """Reset the global timing tracker."""
    global _global_tracker
    _global_tracker = TimingTracker()
    _global_tracker.start_query()


# Convenience functions
def start_timer(key: str):
    """Start a timer with the given key."""
    get_tracker().start_timer(key)


def end_timer(key: str) -> float:
    """End a timer and return elapsed time."""
    return get_tracker().end_timer(key)


def record_agent_start(agent_name: str):
    """Record the start of an agent execution."""
    get_tracker().record_agent_start(agent_name)


def record_agent_end(agent_name: str):
    """Record the end of an agent execution."""
    get_tracker().record_agent_end(agent_name)


def record_llm_call(agent_name: str, duration: float, call_number: int):
    """Record an LLM call for an agent."""
    get_tracker().record_llm_call(agent_name, duration, call_number)


def record_tool_call(agent_name: str, tool_name: str, duration: float):
    """Record a tool call for an agent."""
    get_tracker().record_tool_call(agent_name, tool_name, duration)


def record_delegation(from_agent: str, to_agent: str, duration: float):
    """Record delegation from one agent to another."""
    get_tracker().record_delegation(from_agent, to_agent, duration)


def print_summary():
    """Print a formatted summary of timings."""
    get_tracker().print_summary()


def save_to_file(filepath: str):
    """Save detailed timing data to a JSON file."""
    get_tracker().save_to_file(filepath)
