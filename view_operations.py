#!/usr/bin/env python3
"""
View operations from test results JSON.

This displays each operation showing LLM reasoning + tool execution = total cost.
"""

import json
import sys
from pathlib import Path


def view_operations(filepath: str, question_number: int = None):
    """View operations from test results JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle both test results and timing files
    if "results" in data:
        # Test results file
        results = data["results"]
        if question_number is not None:
            if question_number < 1 or question_number > len(results):
                print(f"Error: Question {question_number} not found (1-{len(results)})")
                return
            results = [results[question_number - 1]]
    elif "detailed_timings" in data:
        # Direct timing file
        results = [{"timing": {"detailed": {"operations": {}}}}]
        # Extract operations from detailed_timings
        for agent_name, agent_data in data["detailed_timings"]["agents"].items():
            if "operations" in agent_data:
                results[0]["timing"]["detailed"]["operations"][agent_name] = agent_data["operations"]
    else:
        results = [data]

    for result_idx, result in enumerate(results, 1):
        if "question_results" in result:
            # This is a test case with multiple questions
            question_results = result["question_results"]
        else:
            # Single result
            question_results = [result]

        for q_idx, q_result in enumerate(question_results, 1):
            question = q_result.get("question", "N/A")
            timing = q_result.get("timing", {})
            operations_data = timing.get("detailed", {}).get("operations", {})

            print(f"\n{'='*80}")
            if len(results) > 1:
                print(f"TEST {result_idx}, QUESTION {q_idx}: {question}")
            else:
                print(f"QUESTION: {question}")
            print(f"{'='*80}")

            if not operations_data:
                print("No operations data found")
                continue

            for agent_name, operations in operations_data.items():
                print(f"\n📊 {agent_name.upper()}")
                print(f"{'-'*80}")

                if not operations:
                    print("  No operations recorded")
                    continue

                # Calculate totals
                total_llm = sum(op["llm_duration"] for op in operations)
                total_tool = sum(
                    sum(t["duration"] for t in op.get("tool_calls", []))
                    for op in operations
                )
                total_cost = sum(op["total_duration"] for op in operations)

                # Display each operation
                for op in operations:
                    op_num = op["operation_number"]
                    llm_duration = op["llm_duration"]
                    tool_calls = op.get("tool_calls", [])
                    total_duration = op["total_duration"]

                    # Operation header
                    if tool_calls:
                        tool_names = ", ".join(t["tool_name"] for t in tool_calls)
                        print(f"\n  Operation #{op_num}: {tool_names}")
                    else:
                        print(f"\n  Operation #{op_num}: Text response (no tools)")

                    # Show cost breakdown
                    print(f"  └─ Total cost: {total_duration:.3f}s")
                    llm_pct = (llm_duration / total_duration * 100) if total_duration > 0 else 0
                    print(f"     ├─ LLM reasoning: {llm_duration:.3f}s ({llm_pct:.1f}%)")

                    if tool_calls:
                        for i, tool in enumerate(tool_calls):
                            is_last = (i == len(tool_calls) - 1)
                            prefix = "└─" if is_last else "├─"
                            tool_duration = tool["duration"]
                            tool_pct = (tool_duration / total_duration * 100) if total_duration > 0 else 0
                            print(f"     {prefix} {tool['tool_name']}: {tool_duration:.3f}s ({tool_pct:.1f}%)")
                    else:
                        print(f"     └─ Tool execution: 0.000s (0.0%)")

                # Summary for this agent
                print(f"\n  {'-'*76}")
                print(f"  SUMMARY: {len(operations)} operations, {total_cost:.3f}s total")
                print(f"    ├─ LLM reasoning: {total_llm:.3f}s ({total_llm/total_cost*100:.1f}%)")
                print(f"    └─ Tool execution: {total_tool:.3f}s ({total_tool/total_cost*100:.1f}%)")

    print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 view_operations.py <test_results.json> [--question N]")
        print("\nExamples:")
        print("  python3 view_operations.py test_results/summary_20260215_212629.json")
        print("  python3 view_operations.py test_results/summary_20260215_212629.json --question 1")
        print("  python3 view_operations.py tmp/timing/timing_20260215_212629.json")
        sys.exit(1)

    filepath = sys.argv[1]

    # Parse optional --question argument
    question_number = None
    if "--question" in sys.argv:
        try:
            question_number = int(sys.argv[sys.argv.index("--question") + 1])
        except (IndexError, ValueError):
            print("Error: --question requires a question number")
            sys.exit(1)

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    view_operations(filepath, question_number)


if __name__ == "__main__":
    main()
