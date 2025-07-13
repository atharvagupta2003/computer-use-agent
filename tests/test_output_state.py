#!/usr/bin/env python3
"""
Test script to verify that the new OutputState works correctly.
"""

import asyncio
import time
from agent.state import ExecutionState, OutputState, TaskStatus, GraphInput
from agent.graph import create_final_output, create_graph

def test_output_state_creation():
    """Test creating OutputState from ExecutionState."""
    print("Testing OutputState creation...")
    
    # Create a completed execution state
    exec_state = ExecutionState(
        status=TaskStatus.completed,
        message_for_user="Found the top 10 Indian dishes: 1. Biryani, 2. Butter Chicken, 3. Samosas, 4. Dosas, 5. Curry, 6. Tandoori Chicken, 7. Naan, 8. Chole Bhature, 9. Rajma, 10. Palak Paneer",
        user_request="find top 10 food dishes in India",
        interaction_count=8,
        action_history=["screenshot", "click_element:search box", "type_text:top 10 food dishes in India", "press_key:enter"],
        screenshots=["screenshot1", "screenshot2", "screenshot3"]
    )
    
    # Create output state
    output_state = OutputState.from_execution_state(exec_state, execution_time=12.5)
    
    print(f"Output State:")
    print(f"  Status: {output_state.status}")
    print(f"  Task Completed: {output_state.task_completed_successfully}")
    print(f"  Answer: {output_state.answer}")
    print(f"  User Request: {output_state.user_request}")
    print(f"  Total Interactions: {output_state.total_interactions}")
    print(f"  Execution Time: {output_state.execution_time_seconds}s")
    print(f"  Screenshots Taken: {output_state.screenshots_taken}")
    print(f"  Actions Performed: {output_state.actions_performed}")
    
    # Test formatted output
    print("\nFormatted Output:")
    print(output_state.get_formatted_result())
    
    # Verify fields
    assert output_state.status == TaskStatus.completed
    assert output_state.task_completed_successfully == True
    assert "Biryani" in output_state.answer
    assert output_state.total_interactions == 8
    assert output_state.execution_time_seconds == 12.5
    assert output_state.screenshots_taken == 3
    assert len(output_state.actions_performed) == 4
    
    print("✓ OutputState creation test passed")
    return True

def test_failed_output_state():
    """Test creating OutputState for failed execution."""
    print("\nTesting failed OutputState creation...")
    
    # Create a failed execution state
    exec_state = ExecutionState(
        status=TaskStatus.failed,
        message_for_user="Failed to connect to sandbox",
        user_request="find top 10 food dishes in India",
        interaction_count=2
    )
    
    # Create output state
    output_state = OutputState.from_execution_state(exec_state, execution_time=5.0)
    
    print(f"Failed Output State:")
    print(f"  Status: {output_state.status}")
    print(f"  Task Completed: {output_state.task_completed_successfully}")
    print(f"  Answer: {output_state.answer}")
    print(f"  Error Message: {output_state.error_message}")
    
    # Test formatted output
    print("\nFormatted Failed Output:")
    print(output_state.get_formatted_result())
    
    # Verify fields
    assert output_state.status == TaskStatus.failed
    assert output_state.task_completed_successfully == False
    assert "Failed to connect" in output_state.answer
    assert output_state.error_message == "Failed to connect to sandbox"
    
    print("✓ Failed OutputState creation test passed")
    return True

async def test_final_output_node():
    """Test the final output node function."""
    print("\nTesting final output node...")
    
    # Create a completed execution state with timing
    exec_state = ExecutionState(
        status=TaskStatus.completed,
        message_for_user="Task completed successfully with results",
        user_request="test request",
        interaction_count=5,
        _start_time=time.time() - 10.0  # 10 seconds ago
    )
    
    # Test the final output node
    output_state = await create_final_output(exec_state)
    
    print(f"Final Output Node Result:")
    print(f"  Type: {type(output_state)}")
    print(f"  Status: {output_state.status}")
    print(f"  Answer: {output_state.answer}")
    print(f"  Execution Time: {output_state.execution_time_seconds:.1f}s")
    
    # Verify it's an OutputState
    assert isinstance(output_state, OutputState)
    assert output_state.status == TaskStatus.completed
    assert output_state.execution_time_seconds >= 0  # Should be >= 0 (could be 0 if no start time)
    
    print("✓ Final output node test passed")
    return True

def test_graph_structure():
    """Test that the graph has the correct structure with final output."""
    print("\nTesting graph structure with final output...")
    
    # Create the graph
    graph = create_graph()
    
    # Check nodes
    expected_nodes = {"setup_sandbox", "brain", "executor", "cleanup", "final_output"}
    actual_nodes = set(graph.nodes.keys())
    
    print(f"Expected nodes: {expected_nodes}")
    print(f"Actual nodes: {actual_nodes}")
    
    if expected_nodes.issubset(actual_nodes):
        print("✓ All expected nodes are present")
    else:
        missing = expected_nodes - actual_nodes
        print(f"✗ Missing nodes: {missing}")
        return False
    
    # Check edges
    edges = graph.edges
    print(f"Graph edges: {edges}")
    
    # Check that final_output -> END exists
    final_to_end = any(edge[0] == "final_output" and edge[1] == "__end__" for edge in edges)
    if final_to_end:
        print("✓ final_output node properly connects to END")
    else:
        print("✗ final_output node does not connect to END")
        return False
    
    # Check that cleanup -> final_output exists
    cleanup_to_final = any(edge[0] == "cleanup" and edge[1] == "final_output" for edge in edges)
    if cleanup_to_final:
        print("✓ cleanup node properly connects to final_output")
    else:
        print("✗ cleanup node does not connect to final_output")
        return False
    
    print("✓ Graph structure test passed")
    return True

def demonstrate_usage():
    """Demonstrate how to use the new OutputState."""
    print("\nDemonstrating OutputState usage...")
    
    usage_example = '''
# Example usage with OutputState:

async def run_agent_with_output(user_request: str):
    from agent.graph import workflow_graph
    from agent.state import GraphInput
    
    # Create input
    input_data = GraphInput(user_request=user_request)
    
    # Run the workflow - now returns OutputState
    result = await workflow_graph.ainvoke(input_data)
    
    # result is now an OutputState with structured information
    print(f"Task Completed: {result.task_completed_successfully}")
    print(f"Answer: {result.answer}")
    print(f"Execution Time: {result.execution_time_seconds:.1f}s")
    print(f"Total Interactions: {result.total_interactions}")
    
    # Get formatted result
    print(result.get_formatted_result())
    
    return result

# Usage:
# output = await run_agent_with_output("find top 10 food dishes in India")
# if output.task_completed_successfully:
#     print("Success:", output.answer)
# else:
#     print("Failed:", output.error_message)
'''
    
    print(usage_example)
    return True

def main():
    """Run all tests."""
    print("Testing OutputState Implementation")
    print("=" * 50)
    
    tests = [
        test_output_state_creation,
        test_failed_output_state,
        lambda: asyncio.run(test_final_output_node()),
        test_graph_structure,
        demonstrate_usage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! OutputState implementation is working correctly.")
        print("\nKey benefits of OutputState:")
        print("- Structured final result with clear success/failure indication")
        print("- Detailed execution metadata (time, interactions, etc.)")
        print("- Formatted output for easy display")
        print("- Separate answer field for the actual result")
        print("- Error handling with specific error messages")
    else:
        print("✗ Some tests failed. OutputState implementation needs fixes.")

if __name__ == "__main__":
    main() 